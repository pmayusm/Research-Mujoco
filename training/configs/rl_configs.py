"""RSL-RL training configurations for the flywheel shooter task."""

GAUSSIAN_DISTRIBUTION = {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
    # Actions are clipped to [-1, 1], so std has no business growing much past that
    # range. Without this cap, the entropy bonus can outweigh our (currently small)
    # task reward and drive std unbounded -- turning the "policy" into clipped noise.
    "std_range": (0.05, 1.5),
}

MLP_ACTOR = {
    "class_name": "MLPModel",
    "hidden_dims": [256, 128, 64],
    "activation": "elu",
    "obs_normalization": True,
    "distribution_cfg": GAUSSIAN_DISTRIBUTION,
}

MLP_CRITIC = {
    "class_name": "MLPModel",
    "hidden_dims": [256, 128, 64],
    "activation": "elu",
    "obs_normalization": True,
}

MLP_TEACHER = {
    "class_name": "MLPModel",
    "hidden_dims": [256, 128, 64],
    "activation": "elu",
    "obs_normalization": True,
}

MLP_STUDENT = {
    "class_name": "MLPModel",
    "hidden_dims": [128, 64],
    "activation": "elu",
    "obs_normalization": True,
    "distribution_cfg": GAUSSIAN_DISTRIBUTION,
}


def teacher_ppo_cfg() -> dict:
    """Privileged teacher policy trained with PPO."""
    return {
        "num_steps_per_env": 24,
        "save_interval": 100,
        "check_for_nan": True,
        "logger": "tensorboard",
        "obs_groups": {
            "actor": ["policy", "privileged"],
            "critic": ["policy", "privileged"],
        },
        "algorithm": {
            "class_name": "PPO",
            "optimizer": "adam",
            "learning_rate": 3e-4,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            # "adaptive" ramps the LR up to 10x whenever KL stays low, which can snowball
            # into a destabilizing update once the policy is fairly confident -- that's the
            # likely cause of the sudden, sustained policy collapse we saw mid-training.
            # A fixed LR is slower but far more predictable for this low-reward-magnitude task.
            "schedule": "fixed",
            "desired_kl": 0.01,
            "entropy_coef": 0.002,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "clip_param": 0.2,
            "use_clipped_value_loss": True,
            "max_grad_norm": 1.0,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": MLP_ACTOR.copy(),
        "critic": MLP_CRITIC.copy(),
    }


def student_distillation_cfg() -> dict:
    """Student policy distilled from the trained teacher."""
    return {
        "num_steps_per_env": 24,
        "save_interval": 100,
        "check_for_nan": True,
        "logger": "tensorboard",
        "obs_groups": {
            "student": ["policy"],
            "teacher": ["policy", "privileged"],
        },
        "algorithm": {
            "class_name": "Distillation",
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "num_learning_epochs": 1,
            "gradient_length": 15,
            "loss_type": "mse",
            "max_grad_norm": 1.0,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "student": MLP_STUDENT.copy(),
        "teacher": MLP_TEACHER.copy(),
    }
