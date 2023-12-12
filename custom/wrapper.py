import csv
import json
import os
import time
from glob import glob
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import pandas
from gymnasium.core import ActType, ObsType


class ArgumentWrapper(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    def __init__(self, env: gym.Env, ns3Settings: dict()):
        super().__init__(env=env)
        self.ns3Settings = ns3Settings

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :return: the first observation of the environment
        """
        return self.env.reset(**kwargs)
