import numpy as np
import json
import random
import math
import time
import vrp_env
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from lib.utils_eval import read_input, create_batch_env, roll_out
from lib.egate_model import Model
from arguments import args
import argparse

if __name__ == "__main__":
    args = args()

    device = torch.device(args.device)

    N_JOBS = int(args.N_JOBS)
    CAP = int(args.CAP)
    batch_size = int(args.BATCH)
    MAX_COORD = int(args.MAX_COORD)
    MAX_DIST = float(args.MAX_DIST)
    LR = float(args.LR)
    DEPOT_END = int(args.DEPOT_END)
    SERVICE_TIME = int(args.SERVICE_TIME)
    TW_WIDTH = int(args.TW_WIDTH)

    N_ROLLOUT = int(args.N_ROLLOUT)
    ROLLOUT_STEPS = int(args.ROLLOUT_STEPS)
    N_STEPS = int(args.N_STEPS)

    model = Model(8,64,2,16)
    model = model.to(device)
    model.load_state_dict(torch.load("model/v8-tw-iter200-rm25-latest.model"))

    inputs = read_input("data/vrptw_99.npy")

    def eval(batch_size, n_steps=100, instance=None):
        envs = create_batch_env(batch_size, 99, instance=instance)
        states = envs.reset()
        states, history, actions, values = roll_out(model,envs,states,n_steps,False,64)
        best_index = np.argmin([env.best for env in envs.envs])
        best_cost = envs.envs[best_index].best
        best_tours = envs.envs[best_index].best_sol
        print ("best cost:", best_cost)
        print ("best tours:", best_tours)
        return best_cost, np.array(history), actions, values

    ave_cost = []
    for i, [data, raw] in enumerate(inputs):
        print("instance " + str(i))
        cost, _, _, _ = eval(batch_size=batch_size, n_steps=N_STEPS, instance=[data, raw])
        ave_cost.append(cost)
    print("ave_cost", np.mean(ave_cost))
