from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class Ns3EnvWrapped(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """Change ns3settings after each reset

    Args:
        env (gym.Env): game environment
    """

    def __init__(
        self,
        env: gym.Env,
    ):
        super().__init__(env=env)

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Change ns3settings seed after reset
        """
        # todo: change how things are accessed
        self.env.unwrapped.ns3Settings["simSeed"] += 1

        return self.env.reset(**kwargs)
