import argparse
import logging
import os
from datetime import datetime

import torch
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3 import A2C
from stable_baselines3.common import logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from tensorboard import program

import ns3ai_gym_env
import gymnasium as gym

from custom.sim_args import TcpRlSimArgs
from custom.model import *
from custom.wrapper import CustomMonitor
from custom.callback import TensorboardCallback
from custom.environment import Ns3EnvWrapped
from custom.env_param_generator import EnvParamGenerator


def run_tensorboard(path: str | os.PathLike) -> None:
    if path:
        tb = program.TensorBoard()
        tb.configure(argv=[None, "--logdir", path])
        url = tb.launch()
        logging.info(f"Tensorflow listening on {url}")


def parse_args() -> argparse.Namespace:
    """Parse input args to object returned.

    Returns:
        argparse.Namespace: object with parsed attributes
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", default=42, type=int, help="set seed for reproducibility"
    )
    parser.add_argument(
        "--input_yaml",
        type=str,
        default=None,
        help="YAML file with simulation parameters",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Location where to save data",
    )
    parser.add_argument(
        "--load_model", default=None, type=str, help="Load model from file, optional"
    )
    parser.add_argument(
        "--tensorboard_log",
        default=True,
        action="store_true",
        help="Log Tensorboard, optional",
    )
    parser.add_argument(
        "--run_tensorboard",
        default=False,
        action="store_true",
        help="Run Tensorboard, optional",
    )
    parser.add_argument(
        "--log_level",
        default="info",
        type=str.upper,
        choices=["CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"],
        help="Log level (CRITICAL, FATAL, ERROR, WARN, WARNING, INFO, DEBUG)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging._nameToLevel[args.log_level])

    output_dir = os.path.join(
        args.output_dir,
        datetime.strftime(datetime.now(), "A2C_%Y-%m-%d_%H-%M-%S"),
    )

    my_seed = args.seed
    logging.info(f"Python side random seed {my_seed}")
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)

    # create env with env_kwargs passed as ns3 arguments
    generator = EnvParamGenerator(TcpRlSimArgs, args.input_yaml)
    env_kwargs = {
        "ns3Settings": generator.generate(),
    }

    env = gym.make(
        "ns3ai_gym_env/Ns3-v0", targetName="rl_tcp_gym", ns3Path="../../", **env_kwargs # type: ignore
    )
    ob_space = env.observation_space
    ac_space = env.action_space
    logging.info(f"Observation space: {ob_space}, {ob_space.dtype}")
    logging.info(f"Action space: {ac_space}, {ac_space.dtype}")

    # wrap env so seed will be changed after each reset
    env = Ns3EnvWrapped(env=env, generator=generator) # type: ignore

    # currently fails as -1 can be returned, fix later
    # check_env(env)

    if args.tensorboard_log:
        tensorboard_log_dir = output_dir
    else:
        tensorboard_log_dir = None

    if args.run_tensorboard:
        run_tensorboard(os.path.dirname(args.output_dir))

    # save each SAR
    env = CustomMonitor(env, filename=os.path.join(output_dir, "monitor.csv"))

    try:
        model = A2C(
            CustomActorCriticPolicy,
            env=env,
            tensorboard_log=tensorboard_log_dir,
            device="cpu",
            verbose=1,
            learning_rate=0.0007,
        )

        if args.load_model:
            model = model.load(args.load_model)

        if tensorboard_log_dir:
            new_logger = logger.configure(output_dir, ["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)
            custom_callback = TensorboardCallback()
        else:
            custom_callback = None

        # print network architecture
        logging.info(model.policy)

        total_timesteps = 100000
        model.learn(
            total_timesteps,
            progress_bar=False,
            log_interval=1,
            callback=custom_callback,
        )

        # manually close env, by default it does not happen
        try:
            model.env.close()
        except UserWarning:
            pass

        # from stable_baselines3.common.evaluation import evaluate_policy
        # evaluate_policy(model, env, n_eval_episodes=1)

        saved_model_path = os.path.join(output_dir, f"model_{total_timesteps}")
        model.save(saved_model_path)

        # del model

        # # check if model loading works
        # model = A2C.load(saved_model_path, env=env)

    # # general catch, do not know all possibilities
    # except Exception as e:
    #     print(e)
    finally:
        logging.info("Finally exiting...")
        # if crashed then close gently
        try:
            env.close()
        except AttributeError:
            pass

    if args.run_tensorboard:
        input("Press enter to close tensorboard.")


if __name__ == "__main__":
    main()
