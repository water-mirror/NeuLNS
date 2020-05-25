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
    DEPOT_END = int(args.DEPOT_END)
    SERVICE_TIME = int(args.SERVICE_TIME)
    TW_WIDTH = int(args.TW_WIDTH)

    envs = create_batch_env(128,99)

    model = Model(8,64,2,16)
    model = model.to(device)

    # model.load_state_dict(torch.load("model/v8-tw-iter200-rm25-latest.model"))
    train(model,envs,1000,20,10,4,n_remove=10)
    torch.save(model.state_dict(), "model/v8-tw-iter200-rm25-latest.model")
