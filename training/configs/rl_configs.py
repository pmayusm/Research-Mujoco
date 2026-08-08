"""RSL-RL training configurations for the flywheel shooter task."""

GAUSSIAN_DISTRIBUTION = {
    "class_name": "GaussianDistribution",
    # Was 1.0: with actions clipped to [-1, 1], std=1 makes every sample nearly
    # uniform over the full control range, so the policy never settles into a
    # precise launch. The gear-fix run kept Mean action std pinned at 1.00 for
    # the entire 30k iterations. Starting lower gives exploration room without
    # drowning the mean action in noise.
    "init_std": 0.5,
    "std_type": "scalar",
    # Soft floor at 0.25: overnight hit peak 14.6% with floor ~0% under std=0.35.
    # Miss still ~74% — precision needs a tighter launch; floor is solved so we
    # can drop exploration without reopening under-launch collapse.
    "std_range": (0.25, 1.0),
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
        # Single main policy: "privileged" is just the target-position channels
        # (not a separate student/teacher setup). Actor sees full state including target.
        "obs_groups": {
            "actor": ["policy", "privileged"],
            "critic": ["policy", "privileged"],
        },
        "algorithm": {
            "class_name": "PPO",
            "optimizer": "adam",
            # Dropped from 3e-4 -> 1e-4: with schedule="fixed", rsl_rl's PPO completely
            # ignores desired_kl (it's only read under "adaptive"), so there is *no*
            # mechanism to correct an update once it starts drifting large. During a
            # resumed run, surrogate loss climbed steadily from ~0.02 to a sustained
            # 0.15-0.35 over ~9k iterations (clip_param is 0.2, so this is well above
            # a healthy, well-clipped update) with nothing to rein it in, and the
            # policy eventually collapsed to 100% floor-drops. A smaller fixed LR keeps
            # each update small enough that this kind of unchecked drift is much less
            # likely, especially once std has already dropped low (0.5ish) and the
            # policy is more sensitive to same-sized absolute steps.
            #
            # Follow-up #1: after the divergence was fixed, a 100k-iteration run held
            # perfectly stable (surrogate loss stayed near 0) but hit rate was flat at
            # ~5% for the entire run -- no upward trend in any of 10 equal-sized
            # segments. Looked like the LR was too conservative to keep progressing.
            #
            # Follow-up #2: it wasn't actually 1e-4. rsl_rl's PPO.load() restores the
            # *optimizer's* saved state dict on resume, which includes the LR the
            # checkpoint was saved with -- silently overriding this config value. That
            # 100k-iteration "1e-4" run, and every resumed run since the value was first
            # dropped from 3e-4, was actually still training at 3e-4 the whole time; the
            # edit here never took effect. (train_teacher.py now explicitly re-applies
            # this config's LR to the optimizer after a resume, so it finally does.)
            # Dropped 3e-4 -> 1e-4 after the gear=1000 run: with std pinned at its
            # floor and schedule="fixed", a late surrogate-loss spike (0.12 then
            # 0.29, above clip_param=0.2) had nothing to rein it in and collapsed a
            # healthy ~6%-hit policy to 100% floor-drops around iter 21k. Smaller
            # fixed steps keep updates inside the trust region when the policy is
            # already sharp.
            # Dropped 1e-4 -> 5e-5 after v8: larger updates + high exploration
            # eroded the good model_202999 mean (hit ~10% -> ~5%). Gentler fixed
            # steps protect a sharp launch policy while still allowing slow progress.
            "learning_rate": 5e-5,
            # Cut 5 -> 3: fewer passes per rollout reduce how hard one bad batch
            # can overwrite a stable launch mean.
            "num_learning_epochs": 3,
            "num_mini_batches": 4,
            # "adaptive" ramps the LR up to 10x whenever KL stays low, which can snowball
            # into a destabilizing update once the policy is fairly confident -- that's the
            # likely cause of the sudden, sustained policy collapse we saw mid-training.
            # A fixed LR is slower but far more predictable for this low-reward-magnitude task.
            "schedule": "fixed",
            "desired_kl": 0.01,
            # Cut 0.0008 -> 0.0003 with std floor 0.25: favor exploiting a sharp
            # launch mean over keeping exploration open (floor already solved).
            "entropy_coef": 0.0003,
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
