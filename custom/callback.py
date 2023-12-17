from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.cum_rew = 0

    def _on_rollout_end(self) -> None:
        self.logger.record("rollout/cum_rew", self.cum_rew)

        # reset vars once recorded
        self.cum_rew = 0
    
    def _on_step(self) -> bool:
        self.cum_rew += self.training_env.get_attr("reward")[0]
        return True