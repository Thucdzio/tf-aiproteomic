import torch
import torch.nn as nn
from model.building_block import *

class ModelMS2Transformer(nn.Module):
    def __init__(   
        self,
        num_frag_types = 8,
        num_modloss_types=4,
        hidden=256,
        dropout=0.1,
        nlayers=4,
        mask_modloss=True,
        max_len=200
    ):
        super().__init__()

        self.num_modloss = num_modloss_types
        self.num_non_modloss = num_frag_types - num_modloss_types
        self.mask_modloss = mask_modloss

        meta_dim = 8

        # ==== Input embedding ====
        self.input_seq_nn = Input_26AA_Mod_PositionalEncoding(
            out_features=hidden - meta_dim, 
            max_len=max_len
        )
        self.meta_nn = Meta_Embedding(out_features=meta_dim)
        

        # ==== Transformer backbone ====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=nlayers
        )

        self.dropout = nn.Dropout(dropout)

        # ==== Main head ====
        self.output_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.PReLU(),
            nn.Linear(64, self.num_non_modloss)
        )

        # ==== Modloss branch ====
        if num_modloss_types > 0:
            modloss_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=8,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                batch_first=True
            )

            self.modloss_transformer = nn.TransformerEncoder(
                modloss_layer,
                num_layers=1
            )

            self.modloss_head = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.PReLU(),
                nn.Linear(64, num_modloss_types)
            )
        else:
            self.modloss_transformer = None
            self.modloss_head = None
    def forward(self, aa_idx, mod_x, charge, nce ,instrument_indices):
        B, L = aa_idx.shape

        # ==== AA + Mod embedding ====
        seq_features = self.input_seq_nn(aa_idx, mod_x) # (B, L, 248)
        meta_features = self.meta_nn(charge, nce, instrument_indices) # (B, 8)
        meta_features = meta_features.unsqueeze(1).expand(-1, L, -1) # (B, L, 8)

       

        x = torch.cat([seq_features, meta_features], dim=2)

        # ==== Backbone ====
        hidden = self.transformer(x)

        # residual trick (paper exact)
        hidden = self.dropout(hidden + 0.2 * x)

        # ==== Main head ====
        out_main = self.output_head(hidden)

        # ==== Modloss ====
        if self.num_modloss > 0:
            if self.mask_modloss:
                zeros = torch.zeros(
                    B, L, self.num_modloss,
                    device=x.device
                )
                out = torch.cat([out_main, zeros], dim=2)
            else:
                modloss_x = self.modloss_transformer(x)
                modloss_x = modloss_x + hidden
                modloss_out = self.modloss_head(modloss_x)
                out = torch.cat([out_main, modloss_out], dim=2)
        else:
            out = out_main

        # remove first cleavage position
        return out[:, 3:, :]
print("hello1")