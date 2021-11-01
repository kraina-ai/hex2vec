import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Tuple

class LitAutoEncoder(pl.LightningModule):

    def __init__(self, sizes):
        super().__init__()

        def create_layers(sizes: List[Tuple[int]]) -> nn.Sequential:
            layers = []
            for i, (input_size, output_size) in enumerate(sizes):
                layers.append(nn.Linear(input_size, output_size))
                if i != len(sizes) - 1:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)
        
        encoder_sizes = list(zip(sizes[:-1], sizes[1:]))
        decoder_sizes = [(output_size, input_size) for input_size, output_size in encoder_sizes][::-1]
        print(encoder_sizes)
        print(decoder_sizes)
        
        self.encoder = create_layers(encoder_sizes)
        self.decoder = create_layers(decoder_sizes)


    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        mae = F.l1_loss(x_hat, x)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        mae = F.l1_loss(x_hat, x)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_mae', mae, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
