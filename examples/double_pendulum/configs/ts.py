import ml_collections
import math
import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "double_pendulum"
    wandb.name = "4_256"  # _db
    wandb.tag = None
    config.data_idx = [
        [-0.3, 0, 0, 0],
        # [-0.2, 0,0,0], [-0.1, 0,0,0], [0.1, 0,0,0],
        # [0.2, 0,0,0],
        # [0.3, 0,0,0],
        [0, -0.3, 0, 0],
        # [0, -0.2,0,0], [0, -0.1,0,0], [0, 0.1, 0,0],
        #                [0, 0.2, 0,0], [0, 0.3, 0,0],
        [0, 0, -0.3, 0],
        #      [0, 0, -0.2,0],
        #    [0, 0, -0.1,0], [0, 0, 0.1, 0],
        #    [0, 0, 0.2, 0],
        #    [0, 0, 0.3, 0],
        [0, 0, 0, -0.3],
        #    [0, 0, 0, -0.2],[0, 0, 0, -0.1], [0, 0, 0, 0.1], [0, 0, 0,  0.2], [0, 0, 0,0.3]
    ]

    # Nondimensionalization
    config.nondim = True

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "ModifiedMlpIDP"  # "ModifiedMlpIDP"
    arch.num_layers = 3
    arch.hidden_dim = 256
    arch.out_dim = 4
    arch.activation = "tanh"  # gelu works better than tanh
    arch.periodicity = None
    #     ml_collections.ConfigDict({
    #     'period': (1.0, 2/3),
    #     'axis': (0, 1),
    #     'trainable': (True, False)
    # })
    # None
    arch.fourier_emb = None
    # ml_collections.ConfigDict(
    #     {"embed_scale": 1 , "embed_dim": 128}
    # )
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9  # 0.9
    optim.decay_steps = 5000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 100001  # 100001
    training.batch_size_per_device = 2000  # 8192

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {
            "q0_loss": 1e2,
            "q1_loss": 1e2,
            "q2_loss": 1e2,
            "r_loss": 1e1,
            # "q0_loss": 1e4,
            # "q1_loss": 1e4,
            # "q2_loss": 1e4,
            # "r_loss": 1e1,
        }
    )

    weighting.momentum = 0.9
    weighting.update_every_steps = 1000  # 100 for grad norm and 1000 for ntk

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
    saving.save_every_steps = 50000  # 100001
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models
    config.input_dim = 1

    # Integer for PRNG random seed.
    config.seed = 42

    return config
