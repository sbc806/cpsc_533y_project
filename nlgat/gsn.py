import torch
import torch.nn as nn

from mlp import Mlps
from helper_functions import compute_gram_matrix, eigendecomposition

class GlobalStructureNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.q_mlps = Mlps(1, [1024])
        self.softmax_layer = nn.Softmax(dim=1)
        self.a_mlps = Mlps(1024, [1024])

    def forward(self, x, format="BNC"):
        # print(x.shape)
        # Compute Gram matrix of whole input point cloud
        # Shape of resulting point cloud should be BNN
        if format == "BCN":
            x_gram_matrix = compute_gram_matrix(x)
        else:
            x_gram_matrix = compute_gram_matrix(x.transpose(-2, -1))
        # print(x_gram_matrix.shape)

        # Obtain eigenvectors matrix
        x_eigenvectors = eigendecomposition(x_gram_matrix)
        # print(x_eigenvectors[:,:,0:1].shape)

        # Get three eigenvectors with most significant eigenvalues
        Q_1 = self.q_mlps(x_eigenvectors[:,:,-1].unsqueeze(-1), format="BNC")
        Q_2 = self.q_mlps(x_eigenvectors[:,:,-2].unsqueeze(-1), format="BNC")
        Q_3 = self.q_mlps(x_eigenvectors[:,:,-3].unsqueeze(-1), format="BNC")
        # print(Q_1.shape)
        Q_1 = self.softmax_layer(Q_1)
        Q_2 = self.softmax_layer(Q_2)
        Q_3 = self.softmax_layer(Q_3)

        first_difference = torch.abs(Q_1 - Q_2)
        second_difference = torch.abs(Q_2 - Q_3)
        third_difference = torch.abs(Q_3 - Q_1)

        A_dp = self.a_mlps(first_difference - second_difference - third_difference, format="BNC")
        return A_dp


