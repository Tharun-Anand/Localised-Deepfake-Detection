import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .models.vit import VitEncoder
from .losses import FocalLoss
from sklearn.metrics import f1_score
import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self,norm_layer=nn.LayerNorm, 
                 embed_dim=768, 
                 use_mean_pooling=True, 
                 fc_drop_rate=0.0):
        super().__init__()
        self.video_encoder = VitEncoder(qkv_bias=True)
        self.au_encoder = VitEncoder(qkv_bias=True)
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.clf_head = nn.Linear(768,2)
        self.focal_loss = FocalLoss(alpha=1, gamma=2)

    def forward_features(self,x):
        x_v = self.video_encoder.patch_embedding(x)
        x_au = self.au_encoder.patch_embedding(x)

        x_v = self.video_encoder.pos_embedding(x_v)
        x_au = self.au_encoder.pos_embedding(x_au)

        au_attns =[block(x_au) for block in self.au_encoder.backbone.blocks]
        x = x_v
        for i,block in enumerate(self.video_encoder.backbone.blocks):
            if i==1:
                x = block(x)
                continue
            x = block(x, au_attns[i-1])
        return x

    def forward(self, videos):
        # videos: B * 3 * 16 * 224 * 224
        encodings = self.forward_features(videos)
        x = self.norm(encodings)
        x =  self.fc_norm(x.mean(1))
        logits = self.clf_head(self.fc_dropout(x))
        return logits
    
    def calculate_clf_loss(self, logits,gt):
        gt = gt.long()
        return self.focal_loss(logits, gt)

    
    def training_step(self, batch, batch_idx):
        videos, labels = batch['video'], batch['labels']
        logits= self(videos)
        loss = self.calculate_clf_loss(logits, labels)
        self.log("train/clf_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        videos, labels = batch['video'], batch['labels']
        with torch.no_grad():
            logits = self(videos)
        loss = self.calculate_clf_loss(logits, labels)
        self.log("val/clf_loss", loss, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels) / len(labels)
        f1 = f1_score(labels.cpu(), preds.cpu())
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True)
        self.log("val/TP", torch.sum((preds == 1) & (labels == 1)).float())
        self.log("val/TN", torch.sum((preds == 0) & (labels == 0)).float())
        self.log("val/FP", torch.sum((preds == 1) & (labels == 0)).float())
        self.log("val/FN", torch.sum((preds == 0) & (labels == 1)).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)