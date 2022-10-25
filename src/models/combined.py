import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Tuple


class Combined(pl.LightningModule):
    def __init__(self, encoder_sizes):
        super().__init__()

        def create_layers(sizes: List[Tuple[int]]) -> nn.Sequential:
            layers = []
            for i, (input_size, output_size) in enumerate(sizes):
                linear = nn.Linear(input_size, output_size)
                nn.init.xavier_uniform_(linear.weight)
                layers.append(nn.Linear(input_size, output_size))
                if i != len(sizes) - 1:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        sizes = list(zip(encoder_sizes[:-1], encoder_sizes[1:]))
        decoder_sizes = [
            (output_size, input_size) for input_size, output_size in sizes
        ][::-1]
        self.encoder = create_layers(sizes)
        self.decoder = create_layers(decoder_sizes)

    def forward(self, Xt: torch.Tensor, Xc: torch.Tensor):
        Xt_em = self.encoder(Xt)
        Xc_em = self.encoder(Xc)
        scores = torch.mul(Xt_em, Xc_em).sum(dim=1)
        return scores, Xt_em, Xc_em

    def predict(self, Xt: torch.Tensor, Xc: torch.Tensor):
        probas = F.sigmoid(self(Xt, Xc))
        return probas

    def training_step(self, batch, batch_idx):
        Xt, Xc, Xn, y_pos, y_neg, *_ = batch
        scores_pos, Xt_em, Xc_em = self(Xt, Xc)
        scores_neg, _, Xn_em = self(Xt, Xn)

        Xt_hat = self.decoder(Xt_em)
        Xc_hat = self.decoder(Xc_em)
        Xn_hat = self.decoder(Xn_em)

        prediction_loss = (
            F.binary_cross_entropy_with_logits(scores_pos, y_pos)
            + F.binary_cross_entropy_with_logits(scores_neg, y_neg)
        ) / 2
        reconstruction_loss = (
            F.mse_loss(Xt, Xt_hat) + F.mse_loss(Xc, Xc_hat) + F.mse_loss(Xn, Xn_hat)
        ) / 3
        loss = prediction_loss + reconstruction_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_prediction_loss", prediction_loss, on_step=True, on_epoch=True)
        self.log(
            "train_reconstruction_loss",
            reconstruction_loss,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        Xt, Xc, Xn, y_pos, y_neg, *_ = batch
        scores_pos, Xt_em, Xc_em = self(Xt, Xc)
        scores_neg, _, Xn_em = self(Xt, Xn)

        Xt_hat = self.decoder(Xt_em)
        Xc_hat = self.decoder(Xc_em)
        Xn_hat = self.decoder(Xn_em)

        prediction_loss = (
            F.binary_cross_entropy_with_logits(scores_pos, y_pos)
            + F.binary_cross_entropy_with_logits(scores_neg, y_neg)
        ) / 2
        reconstruction_loss = (
            F.mse_loss(Xt, Xt_hat) + F.mse_loss(Xc, Xc_hat) + F.mse_loss(Xn, Xn_hat)
        ) / 3
        loss = prediction_loss + reconstruction_loss
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_prediction_loss", prediction_loss, on_step=True, on_epoch=True)
        self.log(
            "val_reconstruction_loss", reconstruction_loss, on_step=True, on_epoch=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
