import ml_collections
import math
import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.mode = "train"
    config.data_idx = [1]

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "membrane"
    wandb.name = "6" 
    wandb.tag = None

    # Nondimensionalization
    config.nondim = True

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "ModifiedMlpIDP" #"ModifiedMlpIDP"
    arch.num_layers = 4
    arch.hidden_dim = 256
    arch.out_dim = 4
    arch.activation = "tanh"  # gelu works better than tanh
    arch.periodicity = None

    #None
    arch.fourier_emb = None 
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 5e-4
    optim.decay_rate = 0.9 #0.9
    optim.decay_steps = 1000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 30000
    training.batch_size_per_device = 20000 #8192

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {
            "u_in": 1e5,
            "v_in": 1e5,
            "w_in": 1e5,
            "r": 1e0,
            "w_bc": 5e5,
            "f": 1e1,
            "stick":1e-1,
        }
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 100  # 100 for grad norm and 1000 for ntk

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 4000
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models
    config.input_dim = 2

    # Integer for PRNG random seed.
    config.seed = 42

    return config
