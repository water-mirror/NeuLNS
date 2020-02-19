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

    N_JOBS = args.N_JOBS
    CAP = args.CAP
    MAX_COORD = args.MAX_COORD
    MAX_DIST = args.MAX_DIST
    LR = args.LR
    DEPOT_END = args.DEPOT_END
    SERVICE_TIME = args.SERVICE_TIME
    TW_WIDTH = args.TW_WIDTH

    envs = create_batch_env(128,99)

    model = Model(8,64,2,16)
    model = model.to(device)

    # model.load_state_dict(torch.load("model/v8-tw-iter200-rm25-latest.model"))
    train(model,envs,1000,20,10,4,n_remove=10)
    torch.save(model.state_dict(), "model/v8-tw-iter200-rm25-latest.model")
