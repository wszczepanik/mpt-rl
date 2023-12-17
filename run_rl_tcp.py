# Copyright (c) 2020-2023 Huazhong University of Science and Technology
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Author: Pengyu Liu <eic_lpy@hust.edu.cn>
#         Hao Yin <haoyin@uw.edu>
#         Muyuan Shen <muyuan_shen@hust.edu.cn>

import os
import torch
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from agents import TcpNewRenoAgent, TcpDeepQAgent, TcpQAgent
import ns3ai_gym_env
import gymnasium as gym
import sys
import traceback

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from custom.model import *
from custom.wrapper import CustomMonitor

logging.basicConfig(level=logging.INFO)


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
    parser.add_argument(
        "--show_log", action="store_true", help="whether show observation and action"
    )
    parser.add_argument("--result", action="store_true", help="whether output figures")
    parser.add_argument(
        "--result_dir", type=str, default="./rl_tcp_results", help="output figures path"
    )
    parser.add_argument(
        "--use_rl", action="store_true", help="whether use rl algorithm"
    )
    parser.add_argument(
        "--rl_algo", type=str, default="DeepQ", help="RL Algorithm, Q or DeepQ"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="scratch/reinforcement-learning/",
        help="Location where to save data",
    )
    parser.add_argument(
        "--load_model", default=None, type=str, help="Load model from file, optional"
    )
    parser.add_argument(
        "--tensorboard_log",
        default=None,
        type=str,
        help="Tensorboard log path, optional",
    )

    args = parser.parse_args()

    my_seed = args.seed
    logging.info(f"Python side random seed {my_seed}")
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)

    my_sim_seed = args.sim_seed

    res_list = [
        "ssThresh_l",
        "cWnd_l",
        "segmentsAcked_l",
        "segmentSize_l",
        "bytesInFlight_l",
    ]
    if args.result:
        for res in res_list:
            globals()[res] = []

    # stepIdx = 0

    # create env with env_kwargs passed as ns3 arguments
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
        log_path = os.path.join(args.save_dir, "log/")
        print(log_path)
        env = Monitor(env, filename=log_path)

        if args.load_model:
            model = A2C.load(
                args.load_model, env=env, tensorboard_log=args.tensorboard_log
            )
        else:
            model = A2C(
                CustomActorCriticPolicy,
                env=env,
                tensorboard_log=args.tensorboard_log,
                verbose=2,
            )

        from stable_baselines3.common.logger import configure

        if args.tensorboard_log:
            from tensorboard import program

            tb = program.TensorBoard()
            tb.configure(argv=[None, "--logdir", args.tensorboard_log])
            url = tb.launch()
            print(f"Tensorflow listening on {url}")

        logging.info(model.policy)

        total_timesteps = 1000
        model.learn(total_timesteps, progress_bar=True, log_interval=1)

        # manually close env, by default it does not happen
        model.env.close()

        # from stable_baselines3.common.evaluation import evaluate_policy
        # evaluate_policy(model, env, n_eval_episodes=1)

        saved_model_path = os.path.join(args.save_dir, "model")
        model.save(saved_model_path)

        del model

        model = A2C.load(saved_model_path, env=env)
        input("Press enter to close.")

    finally:
        logging.info("Finally exiting...")
        # if crashed then close gently
        try:
            env.close()
        except AttributeError:
            pass
