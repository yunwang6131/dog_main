from .ppo_est import PPOEst


def __getattr__(name: str):
    if name == "Dreamer":
        from .dreamer import Dreamer

        return Dreamer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
