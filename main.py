# generic imports
import os
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
from collections import namedtuple
from tqdm.auto import tqdm

# torch imports
import torch

# pyg imports
import torch_geometric
from torch_geometric.nn import to_hetero

# pytorch lightning imports
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy, DDPSpawnStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

# framework imports
from data_handling import BasicDataset
from models import GCN

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

AVAIL_GPUS = 1
BATCH_SIZE = 32
N_WORKERS = 0
NUM_EPOCHS = 5

PATH_DATASET = '/mnt/d/waleed/kaggle/IceCube/data/'
batch = pd.read_parquet(os.path.join(PATH_DATASET, "train/batch_1.parquet"))

dataset = BasicDataset(batches=batch)


train_size = int(0.6 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = int(0.3 * len(dataset))

train_size += int(math.fabs((train_size+val_size+test_size)-len(dataset)))

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size])

train_dataloader = torch_geometric.loader.DataLoader(train_dataset, 
                    batch_size=BATCH_SIZE, 
                    shuffle=True, 
                    num_workers=N_WORKERS, 
                    worker_init_fn=seed_worker,
                    generator=g)
val_dataloader = torch_geometric.loader.DataLoader(val_dataset, 
                    batch_size=BATCH_SIZE, 
                    shuffle=False, 
                    num_workers=0)
                    #worker_init_fn=seed_worker,
                    #generator=g)
test_dataloader = torch_geometric.loader.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
print(next(iter(train_dataloader)))
