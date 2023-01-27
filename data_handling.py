# generic imports
import os
import numpy as np
import pandas as pd

# torch imports
import torch
from torch.utils.data import Dataset

# pyg imports
import torch_geometric.data as PyGData

# torch cluster imports
from torch_cluster import knn_graph


class BasicDataset(Dataset):
    def __init__(self, batches, auxiliary=False, k_neighbors=1, scale=None):

        self.PATH_DATASET = '/mnt/d/waleed/kaggle/IceCube/data/'
        if batches is not None:
            self.batches = batches
        else:
            raise TypeError("List of files should contain at least 1 train parquet file")

        self.scale = scale
        self.initialize()

        self.event_ids = np.unique(self.batches.index)
        self.items     = dict(zip(np.arange(self.event_ids.shape[0]), self.event_ids))

        self.auxiliary   = auxiliary
        self.k_neighbors = k_neighbors

    def initialize(self):
        self.meta_data = pd.read_parquet(os.path.join(self.PATH_DATASET, 'train_meta.parquet'))
        self.geometry  = pd.read_csv(os.path.join(self.PATH_DATASET, 'sensor_geometry.csv'))
        if isinstance(self.batches, list):
            self.batches   = pd.concat(self.batches, ignore_index=True)

    def getdata(self):
        df = self.data[self.features]
        x, y = df.iloc[:, 0:-1].to_numpy(), df.iloc[:, -1].to_numpy()
        if self.scale is not None:
            x /= self.scale

        return x, y

    def __getitem__(self, item):
        event_num = self.items.get(item)
        position  = self.geometry.iloc[self.batches.loc[event_num].sensor_id.tolist(), 1:].to_numpy()
        features  = self.batches.loc[event_num,['time', 'charge', 'auxiliary']].astype(float).to_numpy()

        data      = np.hstack([position, features])
        target    = self.meta_data[self.meta_data.event_id==event_num][['azimuth', 'zenith']].to_numpy().flatten()

        if self.auxiliary:
            good_idx = np.where(data[:,-1]==0)[0]
            data = data[good_idx][:, 1:]

        x = torch.tensor(data, dtype=torch.float32)
        y = self.meta_data[self.meta_data.event_id==event_num][['azimuth', 'zenith']].to_numpy().flatten()
        y = torch.tensor(y, dtype=torch.float32)
        # graph connecvtivety by k_nn algorithm
        knn_graph_edge_index = knn_graph(x[:, 0:3], k=self.k_neighbors)

        return {"data": PyGData.Data(x=x, y=y, edge_index=knn_graph_edge_index), "targets": y}

    def __len__(self):
        return self.event_ids.shape[0]
