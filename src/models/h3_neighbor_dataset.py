import h3
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class H3NeighborDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.data_torch = torch.Tensor(self.data.to_numpy())
        all_indices = set(data.index)

        self.inputs = []
        self.contexts = []
        self.input_h3 = []
        self.context_h3 = []

        self.positive_indexes = {}

        for i, (h3_index, hex_data) in tqdm(enumerate(self.data.iterrows()), total=len(self.data)):
            hex_neighbors_h3 = h3.k_ring(h3_index, 1)
            hex_neighbors_h3.remove(h3_index)
            available_neighbors_h3 = list(hex_neighbors_h3.intersection(all_indices))

            contexts_indexes = [self.data.index.get_loc(idx) for idx in available_neighbors_h3]

            negative_excluded_h3 = h3.k_ring(h3_index, 2)
            negative_excluded_h3 = list(negative_excluded_h3.intersection(all_indices))
            positive_indexes = [self.data.index.get_loc(idx) for idx in negative_excluded_h3]

            self.inputs.extend([i] * len(contexts_indexes))
            self.contexts.extend(contexts_indexes)
            self.positive_indexes[h3_index] = set(positive_indexes)

            self.input_h3.extend([h3_index] * len(available_neighbors_h3))
            self.context_h3.extend(available_neighbors_h3)

        self.inputs = np.array(self.inputs)
        self.contexts = np.array(self.contexts)

        self.input_h3 = np.array(self.input_h3)
        self.context_h3 = np.array(self.context_h3)
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = self.data_torch[self.inputs[index]]
        context = self.data_torch[self.contexts[index]]
        input_h3 = self.input_h3[index]
        neg_index = self.get_random_negative_index(input_h3)
        negative = self.data_torch[neg_index]
        y_pos = 1.0
        y_neg = 0.0

        context_h3 = self.context_h3[index]
        negative_h3 = self.data.index[neg_index]
        return input, context, negative, y_pos, y_neg, input_h3, context_h3, negative_h3

    def get_random_negative_index(self, input_h3):
        excluded_indexes = self.positive_indexes[input_h3]
        negative = np.random.randint(0, len(self.data))
        while negative in excluded_indexes:
            negative = np.random.randint(0, len(self.data))
        return negative
