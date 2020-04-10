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

    N_JOBS = args.N_JOBS
    CAP = args.CAP
    MAX_COORD = args.MAX_COORD
    MAX_DIST = args.MAX_DIST
    LR = args.LR
    DEPOT_END = args.DEPOT_END
    SERVICE_TIME = args.SERVICE_TIME
    TW_WIDTH = args.TW_WIDTH

    N_ROLLOUT = args.N_ROLLOUT
    ROLLOUT_STEPS = args.ROLLOUT_STEPS
    N_STEPS = args.N_STEPS

    model = Model(8,64,2,16)
    model = model.to(device)
    model.load_state_dict(torch.load("model/v8-tw-iter200-rm25-latest.model"))

    inputs = read_input("data/vrptw_99.npy")

    def eval(batch_size=128, n_steps=100, instance=None):
        envs = create_batch_env(batch_size, 99, instance=instance)
        states = envs.reset()
        states, history, actions, values = roll_out(model,envs,states,n_steps,False,64)
        best_index = np.argmin([env.best for env in envs.envs])
        best_cost = envs.envs[best_index].best
        best_tours = envs.envs[best_index].best_sol
        print ("best cost:", best_cost)
        print ("best tours:", best_tours)
        return best_cost, np.array(history), actions, values

    for i, [data, raw] in enumerate(inputs):
        print("instance " + str(i))
        _ = eval(batch_size=256, n_steps=N_STEPS, instance=[data, raw])
