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

from custom.model import *
from custom.wrapper import CustomMonitor
from custom.callback import TensorboardCallback


def run_tensorboard(path=str | os.PathLike) -> None:
    if path:
        tb = program.TensorBoard()
        tb.configure(argv=[None, "--logdir", path])
        url = tb.launch()
        logging.info(f"Tensorflow listening on {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", default=42, type=int, help="set seed for reproducibility"
    )
    parser.add_argument(
        "--sim_seed", default=0, type=int, help="set simulation run number"
    )
    parser.add_argument(
        "--duration", type=float, default=10, help="set simulation duration (seconds)"
    )
    parser.add_argument("--result", action="store_true", help="whether output figures")
    parser.add_argument(
        "--result_dir", type=str, default="./rl_tcp_results", help="output figures path"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="save",
        help="Location where to save data",
    )
    parser.add_argument(
        "--load_model", default=None, type=str, help="Load model from file, optional"
    )
    parser.add_argument(
        "--tensorboard_log",
        default=False,
        type=bool,
        help="Tensorboard log path, optional",
    )
    parser.add_argument(
        "--log_level",
        default="info",
        type=str.upper,
        help="Log level (CRITICAL, FATAL, ERROR, WARN, WARNING, INFO, DEBUG)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging._nameToLevel[args.log_level])

    save_dir = os.path.join(
        args.save_dir,
        datetime.strftime(datetime.now(), "A2C_%Y-%m-%d_%H-%M-%S"),
    )

    my_seed = args.seed
    logging.info(f"Python side random seed {my_seed}")
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)

    # res_list = [
    #     "ssThresh_l",
    #     "cWnd_l",
    #     "segmentsAcked_l",
    #     "segmentSize_l",
    #     "bytesInFlight_l",
    # ]
    # if args.result:
    #     for res in res_list:
    #         globals()[res] = []

    # stepIdx = 0

    # create env with env_kwargs passed as ns3 arguments
    my_sim_seed = args.sim_seed
    env_kwargs = {
        "ns3Settings": {
            "transport_prot": "TcpRlTimeBased",
            "duration": args.duration,
            "simSeed": my_sim_seed,
            "envTimeStep": 0.1,
        },
    }
    env = gym.make(
        "ns3ai_gym_env/Ns3-v0", targetName="rl_tcp_gym", ns3Path="../../", **env_kwargs
    )
    ob_space = env.observation_space
    ac_space = env.action_space
    logging.info(f"Observation space: {ob_space}, {ob_space.dtype}")
    logging.info(f"Action space: {ac_space}, {ac_space.dtype}")

    # currently fails as -1 can be returned, fix later
    # check_env(env)

    try:
        if args.tensorboard_log:
            tensorboard_log_dir = save_dir
        else:
            tensorboard_log_dir = None

        env = Monitor(env, filename=os.path.join(save_dir, "monitor.csv"))

        if args.load_model:
            model = A2C.load(
                args.load_model,
                env=env,
                tensorboard_log=tensorboard_log_dir,
                verbose=1,
            )
        else:
            model = A2C(
                CustomActorCriticPolicy,
                env=env,
                tensorboard_log=tensorboard_log_dir,
                verbose=1,
            )

        if tensorboard_log_dir:
            new_logger = logger.configure(save_dir, ["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)
            custom_callback = TensorboardCallback()
            # one directory up to show all trials
            run_tensorboard(os.path.dirname(save_dir))
        else:
            custom_callback = None

        # print network architecture
        logging.info(model.policy)

        total_timesteps = 1000
        model.learn(
            total_timesteps, progress_bar=True, log_interval=1, callback=custom_callback
        )

        # manually close env, by default it does not happen
        try:
            model.env.close()
        except UserWarning:
            pass

        # from stable_baselines3.common.evaluation import evaluate_policy
        # evaluate_policy(model, env, n_eval_episodes=1)

        saved_model_path = os.path.join(args.save_dir, "model")
        model.save(saved_model_path)

        del model

        # check if model loading works
        model = A2C.load(saved_model_path, env=env)

    # general catch, do not know all possibilities
    except Exception as e:
        print(e)
    finally:
        logging.info("Finally exiting...")
        # if crashed then close gently
        try:
            env.close()
        except AttributeError:
            pass

    if args.tensorboard_log:
        input("Press enter to close tensorboard.")
