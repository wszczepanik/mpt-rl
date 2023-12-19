from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """Save additional data for tensorboard

    Args:
        BaseCallback (_type_): _description_
    """

    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.cum_rew = 0

    def _on_step(self) -> bool:
        """Run after each step, collect actions and reward

        Returns:
            bool: _description_
        """
        self.cum_rew += self.training_env.unwrapped.buf_rews[0]  # type: ignore

        self.logger.record("train/cum_rew", self.cum_rew)
        self.logger.record("train/ssThresh", self.training_env.unwrapped.actions[0][0])  # type: ignore
        self.logger.record("train/cWnd", self.training_env.unwrapped.actions[0][1])  # type: ignore

        # reset at episode end
        if any(self.training_env.unwrapped.buf_dones):  # type: ignore
            self.cum_rew = 0

        return True
