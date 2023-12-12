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
from custom.wrapper import ArgumentWrapper

logging.basicConfig(level=logging.INFO)


def get_agent(socketUuid, useRl):
    agent = get_agent.tcpAgents.get(socketUuid)
    if agent is None:
        if useRl:
            if args.rl_algo == "DeepQ":
                agent = TcpDeepQAgent()
                print("new Deep Q-learning agent, uuid = {}".format(socketUuid))
            else:
                agent = TcpQAgent()
                print("new Q-learning agent, uuid = {}".format(socketUuid))
        else:
            agent = TcpNewRenoAgent()
            print("new New Reno agent, uuid = {}".format(socketUuid))
        get_agent.tcpAgents[socketUuid] = agent

    return agent


if __name__ == "__main__":
    # initialize variable
    get_agent.tcpAgents = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="set seed for reproducibility")
    parser.add_argument("--sim_seed", type=int, help="set simulation run number")
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
        "--save_dir", type=str, default="saved/", help="Location where to save data"
    )
    parser.add_argument("--load_model", type=str, help="Load model from file, optional")

    args = parser.parse_args()
    my_seed = 42
    if args.seed:
        my_seed = args.seed
    logging.info(f"Python side random seed {my_seed}")
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)

    my_sim_seed = 0
    if args.sim_seed:
        my_sim_seed = args.sim_seed

    # if args.use_rl:
    #     if (args.rl_algo != 'Q') and (args.rl_algo != 'DeepQ'):
    #         print("Invalid RL Algorithm {}".format(args.rl_algo))
    #         exit(1)

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
        env = ArgumentWrapper(env, env_kwargs)
        env = Monitor(env, filename=log_path)

        if args.load_model:
            model = A2C.load(args.load_model, env=env)
        else:
            model = A2C(CustomActorCriticPolicy, env=env, verbose=2)

        logging.info(model.policy)

        total_timesteps = 1000
        model.learn(total_timesteps, progress_bar=True, log_interval=10)

        # manually close env, by default it does not happen
        model.env.close()

        results_plotter.plot_results(
            [log_path], total_timesteps, results_plotter.X_TIMESTEPS, "test_name"
        )

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

    # try:
    #     obs, info = env.reset()
    #     reward = 0
    #     done = False

    #     model = A2C(CustomActorCriticPolicy, env, verbose=2).learn(
    #         10000, progress_bar=True
    #     )
    #     # model.learn
    #     print(model.policy)

    #     obs, info = env.reset()
    #     for i in range(1000):
    #         print(obs)
    #         action, _state = model.predict(obs, deterministic=True)

    #         print(action)
    #         action = np.rint(action).astype(int)
    #         # action = action.to(torch.int32)
    #         obs, reward, terminated, _truncated, info = env.step(action)

    #         model.train()

    #     # while True:
    #     #     # current ssThreshold
    #     #     ssThresh = obs[4]
    #     #     # current contention window size
    #     #     cWnd = obs[5]
    #     #     # segment size
    #     #     segmentSize = obs[6]
    #     #     # number of acked segments
    #     #     segmentsAcked = obs[9]
    #     #     # estimated bytes in flight
    #     #     bytesInFlight = obs[7]

    #     #     cur_obs = [ssThresh, cWnd, segmentsAcked, segmentSize, bytesInFlight]
    #     #     if args.show_log:
    #     #         logging.info("Recv obs:", cur_obs)

    #     #     if args.result:
    #     #         for res in res_list:
    #     #             globals()[res].append(globals()[res[:-2]])

    #     #     # action = tcpAgent.get_action(obs, reward, done, info)
    #     #     action = 2

    #     #     if args.show_log:
    #     #         logging.info("Step:", stepIdx)
    #     #         stepIdx += 1
    #     #         logging.info("Send act:", action)

    #     #     obs, reward, done, _, info = env.step(action)

    #     #     if done:
    #     #         logging.info("Simulation ended")
    #     #         break

    #     #     # get existing agent of create new TCP agent if needed
    #     #     # tcpAgent = get_agent(obs[0], args.use_rl)

    # except Exception as e:
    #     exc_type, exc_value, exc_traceback = sys.exc_info()
    #     logging.error("Exception occurred: {}".format(e))
    #     logging.error("Traceback:")
    #     traceback.print_tb(exc_traceback)
    #     exit(1)

    # else:
    #     if args.result:
    #         if args.result_dir:
    #             if not os.path.exists(args.result_dir):
    #                 os.mkdir(args.result_dir)
    #         for res in res_list:
    #             y = globals()[res]
    #             x = range(len(y))
    #             plt.clf()
    #             plt.plot(x, y, label=res[:-2], linewidth=1, color="r")
    #             plt.xlabel("Step Number")
    #             plt.title("Information of {}".format(res[:-2]))
    #             plt.savefig("{}.png".format(os.path.join(args.result_dir, res[:-2])))

    # finally:
    #     logging.info("Finally exiting...")
    #     env.close()
