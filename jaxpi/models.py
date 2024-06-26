from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Dict

from flax.training import train_state
from flax import jax_utils

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, pmap, random, tree_map, jacfwd, jacrev
from jax.tree_util import tree_map, tree_reduce, tree_leaves

import optax

from jaxpi import archs
from jaxpi.utils import flatten_pytree
from jax.experimental.host_callback import call

class TrainState(train_state.TrainState):
    weights: Dict
    momentum: float

    def apply_weights(self, weights, **kwargs):
        """Updates `weights` using running average  in return value.

        Returns:
          An updated instance of `self` with new weights updated by applying `running_average`,
          and additional attributes replaced as specified by `kwargs`.
        """

        # Define the running average function
        def running_average(old_w, new_w, momentum):
            return old_w * momentum + new_w * (1 - momentum)

        # Update the weights using running average
        def update_weights(old_weights, new_weights, momentum):
            return tree_map(lambda old_w, new_w: running_average(old_w, new_w, momentum), old_weights, new_weights)

        # Update the weights in place
        weights = update_weights(self.weights, weights, self.momentum)
        return lax.stop_gradient(weights)

        # return self.replace(
        #     step=self.step,
        #     params=self.params,
        #     opt_state=self.opt_state,
        #     weights=weights,
        #     **kwargs,
        # )


def _create_arch(config):
    if config.arch_name == "Mlp":
        arch = archs.Mlp(**config)

    elif config.arch_name == "ModifiedMlp":
        arch = archs.ModifiedMlp(**config)

    elif config.arch_name == "DeepONet":
        arch = archs.DeepONet(**config)

    elif config.arch_name == "ModifiedMlpIDP":
        arch = archs.ModifiedMlpIDP(**config)

    else:
        raise NotImplementedError(f"Arch {config.arch_name} not supported yet!")

    return arch


def _create_optimizer(config):
    if config.optimizer == "Adam":
        lr = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.decay_steps,
            decay_rate=config.decay_rate,
        )
        tx = optax.adam(
            learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps
        )

    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not supported yet!")

    # Gradient accumulation
    if config.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.grad_accum_steps)

    return tx


def _create_train_state(config, trainable_parameters = None):
    # Initialize network
    arch = _create_arch(config.arch)
    x = jnp.ones(config.input_dim)
    params = arch.init(random.PRNGKey(config.seed), x)
    

    if trainable_parameters is not None:
        params["params"]["trainable_parameters"] = trainable_parameters[0]

    # Initialize optax optimizer
    tx = _create_optimizer(config.optim)

    # Convert config dict to dict
    init_weights = dict(config.weighting.init_weights)

    state = TrainState.create(
        apply_fn=arch.apply,
        params=params,
        tx=tx,
        weights=init_weights,
        momentum=config.weighting.momentum,
    )

    return jax_utils.replicate(state)


class PINN:
    def __init__(self, config, trainable_parameters = None):
        self.config = config
        self.state = _create_train_state(config, trainable_parameters)
        self.weights = self.state.weights # TODO: this is a bandage treatment

    def u_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def r_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def losses(self, params, batch, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def compute_diag_ntk(self, params, batch, *args):
        raise NotImplementedError("Subclasses should implement this!")

    @partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch, *args):
        # Compute losses
        losses = self.losses(params, batch, *args)
        # Compute weighted loss
        weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
        # Sum weighted losses
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
        return loss


    @partial(jit, static_argnums=(0,))
    def compute_weights(self, params, batch, *args):
        if self.config.weighting.scheme == "grad_norm":
            # Compute the gradient of each loss w.r.t. the parameters
            grads = jacrev(self.losses)(params, batch, *args)
            grads = lax.stop_gradient(grads)

            # Compute the grad norm of each loss
            grad_norm_dict = {}
            for key, value in grads.items():
                flattened_grad = flatten_pytree(value)
                grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)

            sum_grad_norm = sum(tree_leaves(grad_norm_dict))

            # Count the total number of elements in grad_norm_dict
            num_elements = sum(len(tree_leaves(grad_norm)) for grad_norm in grad_norm_dict.values())

            # Compute the mean grad norm
            mean_grad_norm = sum_grad_norm / num_elements

            # Grad Norm Weighting
            w = tree_map(lambda x: (mean_grad_norm / x), grad_norm_dict)

        elif self.config.weighting.scheme == "ntk":
            # Compute the diagonal of the NTK of each loss
            ntk = self.compute_diag_ntk(params, batch, *args)

            # Compute the mean of the diagonal NTK corresponding to each loss
            mean_ntk_dict = tree_map(lambda x: jnp.mean(x), ntk)

            # Compute the average over all ntk means
            mean_ntk = jnp.mean(jnp.stack(tree_leaves(mean_ntk_dict)))
            # NTK Weighting
            w = tree_map(lambda x: (mean_ntk / x), mean_ntk_dict)

        return w
    
    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def step(self, state, batch, *args):
        # TODO: increase measurement loss 
        grads = grad(self.loss)(state.params, state.weights, batch, *args)
        grads = lax.pmean(grads, "batch")
        state = state.apply_gradients(grads=grads)
        return state
    
    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def update_weights(self, state, batch, *args):
        weights = self.compute_weights(state.params, batch, *args)
        weights = lax.pmean(weights, "batch")
        return state.apply_weights(weights=weights)
        # return 




class ForwardIVP(PINN):
    def __init__(self, config):
        super().__init__(config)

        if config.weighting.use_causal:
            self.tol = config.weighting.causal_tol
            self.num_chunks = config.weighting.num_chunks
            self.M = jnp.triu(jnp.ones((self.num_chunks, self.num_chunks)), k=1).T


class ForwardBVP(PINN):
    def __init__(self, config, **kwarg):
        super().__init__(config, **kwarg)
