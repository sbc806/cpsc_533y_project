import torch
import torch.nn as nn

from mlp import Mlps
from helper_functions import compute_gram_matrix, sort_gram_matrix_rows
from helper_functions import get_knn_idx, index_points
from helper_functions import get_local_gram, get_similar

class GlobalRepresentationNetwork(nn.Module):
    def __init__(self, config):
        super(GlobalRepresentationNetwork, self).__init__()

        self.config = config

        self.localize = config.localize
        self.descending = config.descending
        self.reduce_option = config.reduce_option

        # Need 3 MLP-ST modules
        self.mlp_1_8_points = Mlps(1, [1024], kernel_size=(3, 1), stride=1, padding="same")

        self.mlp_1 = Mlps(3, [1024], kernel_size=(3, 1), stride=1, padding="same")
        self.mlp_2 = Mlps(1024, [1024], kernel_size=1)
        self.mlp_3 = Mlps(1024, [1024])
        self.mlp_4 = Mlps(1024, [1024])

        # Need to set self.network() and self.mlp
        self.mlp_st_1 = MLP_ST(config.num_pts, config.use_cuda)
        self.mlp_st_2 = MLP_ST(config.num_pts, config.use_cuda)
        self.mlp_st_3 = MLP_ST(config.num_pts, config.use_cuda)

        self.network_channels_mlp = Mlps(1024, [1024])
        self.network_channels_mlp_st = MLP_ST(1, config.use_cuda)

    def forward(self, x, format="BNC"):

        # Obtain l, kop2, and kop3
        # Construct three gram matrixces, Nxkkx, where k is kop1, kop2, or kp3
        # optimal_k = construct_matrices(x)
        if format == "BNC":
            x = x.transpose(-2, -1)
        # print(x.shape)

        # Use 3 different k to construct 3 Gram matrices
        # k_1 = 2
        # k_2 = 4
        # k_3 = 8

        if self.config.num_pts == 8:
            knn_idx_1 = get_knn_idx(x, x, 8)
            knn_features_1 = index_points(x, knn_idx_1)
            gram_matrix_1 = compute_gram_matrix(knn_features_1.transpose(2, 1))

            sorted_gram_matrix_1 = sort_gram_matrix_rows(gram_matrix_1, descending=self.descending)
            enhanced_knn_features_1 = self.mlp_st_1(sorted_gram_matrix_1)

            if self.reduce_option == "average:":
                X_hat_j_1 = torch.mean(enhanced_knn_features_1, dim=-2)
            elif self.reduce_option == "absolute_average":
                X_hat_j_1 = torch.mean(torch.abs(enhanced_knn_features_1), dim=-2)
            else:
                X_hat_j_1, _ = torch.max(enhanced_knn_features_1, dim=-2)
            self.max_pool_1 = nn.MaxPool1d(8)
            X_hat_1 = self.max_pool_1(X_hat_j_1)
            X_concatenated = X_hat_1

        else:
            k_1 = self.config.k_1
            k_2 = self.config.k_2
            k_3 = self.config.k_3
            knn_idx_1 = get_knn_idx(x, x, k_1+1)
            knn_idx_2 = get_knn_idx(x, x, k_2+1)
            knn_idx_3 = get_knn_idx(x, x, k_3+1)
            # print(knn_idx_1.shape)
            similar_knn_idx_1, _ = get_similar(x, k_1-1, self.localize)
            similar_knn_idx_2, _ = get_similar(x, k_2-1, self.localize)
            similar_knn_idx_3, _ = get_similar(x, k_3-1, self.localize)

            knn_idx_1 = torch.concatenate((knn_idx_1, similar_knn_idx_1), dim=-1)
            knn_idx_2 = torch.concatenate((knn_idx_2, similar_knn_idx_2), dim=-1)
            knn_idx_3 = torch.concatenate((knn_idx_3, similar_knn_idx_3), dim=-1)
            # print(knn_idx_1.shape,knn_idx_2.shape,knn_idx_3.shape)
            knn_features_1 = index_points(x, knn_idx_1)
            knn_features_2 = index_points(x, knn_idx_2)
            knn_features_3 = index_points(x, knn_idx_3)

            # print("knn_features.shape:", knn_features_1.shape,knn_features_2.shape)
            # Run through MLP-ST module
            gram_matrix_1 = compute_gram_matrix(knn_features_1.transpose(2, 1))
            gram_matrix_2 = compute_gram_matrix(knn_features_2.transpose(2, 1))
            gram_matrix_3 = compute_gram_matrix(knn_features_3.transpose(2, 1))
            # print("gram_matrix_1.shape:", gram_matrix_1.shape)

            # Sort rows of the Gram matrix
            sorted_gram_matrix_1 = sort_gram_matrix_rows(gram_matrix_1, descending=self.descending)
            sorted_gram_matrix_2 = sort_gram_matrix_rows(gram_matrix_2, descending=self.descending)
            sorted_gram_matrix_3 = sort_gram_matrix_rows(gram_matrix_3, descending=self.descending)
            # print(sorted_gram_matrix_1.shape)
            enhanced_knn_features_1 = self.mlp_st_1(sorted_gram_matrix_1)
            enhanced_knn_features_2 = self.mlp_st_2(sorted_gram_matrix_2)
            enhanced_knn_features_3 = self.mlp_st_3(sorted_gram_matrix_3)
            # print("Finished enhancing")
            # Assume no have 3xNx64
            if self.reduce_option == "average":
                X_hat_j_1 = torch.mean(enhanced_knn_features_1, dim=-2)
                X_hat_j_2 = torch.mean(enhanced_knn_features_2, dim=-2)
                X_hat_j_3 = torch.mean(enhanced_knn_features_3, dim=-2)
            elif self.reduce_option == "absolute_average":
                X_hat_j_1 = torch.mean(torch.abs(enhanced_knn_features_1), dim=-2)
                X_hat_j_2 = torch.mean(torch.abs(enhanced_knn_features_2), dim=-2)
                X_hat_j_3 = torch.mean(torch.abs(enhanced_knn_features_3), dim=-2)
            elif self.reduce_option == "maximum":
                X_hat_j_1, _ = torch.max(enhanced_knn_features_1, dim=-2)
                X_hat_j_2, _ = torch.max(enhanced_knn_features_2, dim=-2)
                X_hat_j_3, _ = torch.max(enhanced_knn_features_3, dim=-2)
            # print(X_hat_j_1.shape)
            # print("Finished reducing")
            # print("X_hat_j_1.shape:", X_hat_j_1.shape)
            # Max pooling so that Nx1
            self.max_pool_1 = nn.MaxPool1d(2*k_1)
            self.max_pool_2 = nn.MaxPool1d(2*k_2)
            self.max_pool_3 = nn.MaxPool1d(2*k_3)
            X_hat_1 = self.max_pool_1(X_hat_j_1)
            X_hat_2 = self.max_pool_2(X_hat_j_2)
            X_hat_3 = self.max_pool_3(X_hat_j_3)
            # Concatenate together so that Nx3
            X_concatenated = torch.concatenate((X_hat_1, X_hat_2, X_hat_3), dim=2)
            # print("X_concatenated.shape:", X_concatenated.shape)

        # Turn to Nx1024
        if self.config.num_pts == 8:
            feature_representation = self.mlp_1_8_points(X_concatenated, format="BNC")
        else:
            feature_representation = self.mlp_1(X_concatenated, format="BNC")
        feature_representation = self.mlp_4(self.mlp_3(self.mlp_2(feature_representation, format="BNC"), format="BNC"), format="BNC")
        # print("feature_representation.shape:", feature_representation.shape)

        channels_fusion_gram_matrix = compute_gram_matrix(feature_representation.transpose(-2, -1)).unsqueeze(1)
        # print(channels_fusion_gram_matrix.shape)
        A_cf = self.network_channels_mlp(feature_representation, format="BNC")
        # print("A_cf.shape:", A_cf.shape)
        X_hat_g = A_cf * feature_representation
        # print("X_hat_g.shape:", X_hat_g.shape)
        return X_hat_g

class MLP_ST(nn.Module):
    def __init__(self, n, use_cuda):
        super(MLP_ST, self).__init__()
        if use_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.mlp = nn.Conv2d(n, n, kernel_size=3, stride=1, padding="same")

        self.conv_1 = nn.Conv2d(n, n, kernel_size=3, stride=1, padding="same")
        self.batch_1 = nn.BatchNorm2d(n)
        self.conv_2 = nn.Conv2d(n, n, kernel_size=3, stride=1, padding="same")
        self.batch_2 = nn.BatchNorm2d(n)

        self.fc_1 = nn.Linear(n, 128)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(128, n)

    def forward(self, x):

        # Assume b, n, k, k

        # From Section 3.3.2

        # Create local feature and feature map
        # MLP without ReLU function
        local_feature = self.mlp(x)

        feature_map = self.conv_1(local_feature.clone())
        feature_map = self.batch_1(feature_map)
        feature_map = self.conv_2(feature_map)
        feature_map = self.batch_2(feature_map)
        # print("feature_map.shape:", feature_map.shape)

        # Create soft threshold and shoft thresholding module
        b, n, m, m = feature_map.shape
        # Compress feature map into one-dimensional vector using absolute value and global average pooling
        self.global_pooling_layer = nn.AvgPool2d(m)
        X_gap = self.global_pooling_layer(torch.abs(feature_map))
        # print("X_gap.shape:", X_gap.shape)

        alphas = self.fc_1(X_gap.squeeze(-1).squeeze(-1))
        alphas = self.relu_1(alphas)
        alphas = self.fc_2(alphas)
        thresholds = alphas.unsqueeze(-1).unsqueeze(-1) * X_gap

        # Local feature activation
        # print("alphas.shape:", alphas.shape)
        # print("thresholds.shape:", thresholds.shape)
        local_feature_activation = torch.sign(feature_map) * torch.where(torch.abs(feature_map) - thresholds > 0,
                                                                           torch.abs(feature_map) - thresholds,
                                                                           0) + local_feature
        # print("local_feature_activation.shape:", local_feature_activation.shape)

        # Local feature enhancement
        X_diag = torch.diagonal(local_feature, dim1=-2, dim2=-1).to(self.device)
        # print(X_diag)
        zeros = torch.zeros((b, n, m, m)).to(self.device)
        X_diag = torch.diagonal_scatter(zeros, X_diag, dim1=-2, dim2=-1).to(self.device)

        # print(X_diag)
        # print("X_diag.shape:",X_diag.shape)
        thresholds = thresholds
        X_diag = torch.sign(X_diag) * torch.where(torch.abs(X_diag) - thresholds > 0,
                                                  torch.abs(X_diag) - thresholds,
                                                  0) + X_diag

        X_diag = X_diag.to(self.device)
        local_feature_enhanced = X_diag * local_feature_activation + torch.diagonal_scatter(local_feature_activation, torch.zeros(b, n, m), dim1=-2, dim2=-1)
        # print("local_feature_enhanced.shape:", local_feature_enhanced.shape)
        return local_feature_enhanced

  