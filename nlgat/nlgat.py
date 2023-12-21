import torch
import torch.nn as nn
import torch.nn.functional as F

from gsn import GlobalStructureNetwork
from grn import GlobalRepresentationNetwork

class NLGAT(nn.Module):
    def __init__(self, config):
        super(NLGAT, self).__init__()

        self.config = config

        self.GSN = GlobalStructureNetwork(config)

        self.GRN = GlobalRepresentationNetwork(config)

        n = config.num_pts
        c = config.num_classes
        self.max_pool=nn.MaxPool1d(n)
        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, c)

    def forward(self, x, format="BNC"):

        if self.config.mode == "GSN" or self.config.mode == "both":
            A_dp = self.GSN(x, format)

        if self.config.mode == "GRN" or self.config.mode == "both":
            X_hat_g = self.GRN(x, format)

        if self.config.mode == "both":
            X_g = A_dp * X_hat_g
        elif self.config.mode == "GSN":
            X_g = A_dp
        elif self.config.mode == "GRN":
            X_g = X_hat_g

        pooled_X_g = self.max_pool(X_g.transpose(2,1)).squeeze(-1)
        # print(f"pooled_X_g.shape: {pooled_X_g.shape}")
        logits = self.fc_3(self.fc_2(self.fc_1(pooled_X_g)))
        # print(f"logits.shape: {logits.shape}")
        return F.log_softmax(logits, dim=1)

    def get_loss(self, pred, label):
        # loss = F.nll_loss(pred, label)
        loss = torch.nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        return loss(pred, label)

    def get_acc(self, pred, label):
        pred_choice = pred.max(dim=1)[1]
        # print(pred_choice.shape)
        # print(label.shape)
        acc = (pred_choice == label).float().mean()
        return acc