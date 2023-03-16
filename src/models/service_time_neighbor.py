import h3
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class H3ServiceNeighbors(Dataset):
    def __init__(self, data: pd.DataFrame, grouped_df: pd.Series):
        self.data = data
        self.data_torch = torch.Tensor(self.data.to_numpy(dtype=np.float32))
        all_indices = set(data.index)

        self.inputs = []
        self.contexts = []
        self.input_h3 = []
        self.context_h3 = []

        self.positive_indexes = {}
        i = 0
        for _, (h3_inds, h3s)  in tqdm(grouped_df.items(), total=len(grouped_df)):
            for h3_index in h3s:
                available_neighbors_h3 = [
                    _h for _h in h3s if _h != h3_index
                ]

                contexts_indexes = [
                    _i for _i, _h in
                    zip(h3_inds, h3s)
                    if _h in available_neighbors_h3
                ]

                # add in the neighbors of h3_index
                neighbors = h3.k_ring(h3_index, 1)
                neighbors = neighbors.difference({h3_index})
                for neighbor in neighbors:
                    if neighbor in all_indices:
                        available_neighbors_h3.append(neighbor)
                        contexts_indexes.append(data.index.get_loc(neighbor))

                self.inputs.extend([i] * len(contexts_indexes))
                self.contexts.extend(contexts_indexes)
                self.positive_indexes[h3_index] = set(contexts_indexes)

                self.input_h3.extend([h3_index] * len(available_neighbors_h3))
                self.context_h3.extend(available_neighbors_h3)
                i += 1

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
