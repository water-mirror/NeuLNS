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
from lib.utils_eval import create_batch_env, roll_out
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
    n_steps=int(args.N_STEPS)

    model = Model(4,64,2,16)
    model = model.to(device)
    envs = create_batch_env()
    model.load_state_dict(torch.load('model/v5-500.model'))

    states = envs.reset()
    print ("before mean cost:",np.mean([env.cost for env in envs.envs]))
    _,states,history = roll_out(model,envs,states,n_steps)
    print ("after mean cost:",np.mean([env.cost for env in envs.envs]))

    history = np.array(history)

    a = [env.env.tours() for env in envs.envs]
    for i in range(len(a)):
        print(i)
        print(a[i])
