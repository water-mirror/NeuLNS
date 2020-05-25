import numpy as np
import json
import random
import math
import vrp_env
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from lib.utils_train import create_batch_env, train
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

    envs = create_batch_env(128,99)

    model = Model(4,64,2,16)
    model = model.to(device)

    train(model,envs,1000,10,10,4)
    torch.save(model.state_dict(), "model/v5-latest.model")
