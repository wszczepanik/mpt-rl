from typing import Any, Dict, Tuple, Type

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class Ns3EnvWrapped(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """Change ns3settings after each reset

    Args:
        env (gym.Env): game environment
        generator (Type): parameter generator class
    """

    def __init__(self, env: gym.Env, generator: Type):
        self.generator = generator
        super().__init__(env=env)

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Change ns3settings parameters after reset
        """

        self.env.unwrapped.ns3Settings = self.generator.generate() # type: ignore

        return self.env.reset(**kwargs)
