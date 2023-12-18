from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class Ns3EnvWrapped(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """Change ns3settings after each reset

    Args:
        gym (_type_): _description_
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
        self.env.ns3Settings["simSeed"] += 1

        return self.env.reset(**kwargs)
