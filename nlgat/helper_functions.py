import torch

def compute_gram_matrix(X):
    """
    Expects the last two dimensions of X to be C x N where C is the dimension of the point cloud
    and N is the number of points in the point cloud.
    """
    # return X @ X.transpose(2, 1)
    return X.transpose(-2, -1) @ X

def sort_gram_matrix_rows(G, descending=False):
    return torch.sort(G, dim=-1, descending=descending)[0]

def eigendecomposition(X):
    # Use torch.linalg.eigh() since X is a Gram matrix, it is symmetric
    # Eigenvalues returned in ascending order and columns are eigenvectors
    # torch.linalg.eigh() is faster than torch.linalg.eig() and for Hermitian
    # and symmetric matrices
    eigenvalues, eigenvectors = torch.linalg.eigh(X)

    real_eigenvectors = eigenvectors.type(torch.float32)
    return real_eigenvectors

def get_local_gram(p, k, localize=True):
    """
    Expects p to be b, c, n.
    Obtains local gram matrices for each point in the point cloud.
    The local gram matrix is formed from the point itself and its k-nearest neighbors.
    Returns local gram matrices, matrix of k-nearest neighbors, and indices of k-nearest neighbors.
    local_gram is b, n, k, k, p_neighborhood is b, n, c, k, and knn_idx is b, n, k.
    """
    b, c, n = p.shape

    # Obtain features of k-nearest neighbors including point itself
    knn_idx = get_knn_idx(p, p, k)
    p_neighborhood = index_points(p, knn_idx)
    # print(p_neighborhood[0,:,1,:])
    # print(p_neighborhood.shape)
    # print(p_neighborhood[0,:,0,:])
    # print(p_neighborhood.mean(dim=-1,keepdim=True)[0,:,0,:])
    if localize:
        p_neighborhood = p_neighborhood - p_neighborhood.mean(dim=-1, keepdim=True)
        # print("Localizing")
    # Reshape b, c, n, k tensor to be b, n, k, c
    p_neighborhood = p_neighborhood.transpose(2, 1)
    p_neighborhood = p_neighborhood.transpose(-1, -2)

    # print(p_neighborhood.shape)
    # print(p_neighborhood[0,1,:,:])
    # print(p_neighborhood[0,1,:,:].transpose(-2,-1))
    local_gram = p_neighborhood @ p_neighborhood.transpose(-2, -1)

    return local_gram, p_neighborhood.transpose(-1, -2), knn_idx

def get_similar(p, k_neighbors, localize=True, compute_average_similar=False):
    """
    Expects p to be b, c, n.
    k_neighbors: The number of smilar points to find.
    Gets all the indices of similar points as a b, n, k_neighbors tensor.
    Avoids double counting the point itself and points in the first-order neighborhood.
    These similar points should satisfy equation 3.
    """
    local_gram_matrix, p_neighborhood, knn_idx = get_local_gram(p, k_neighbors, localize)

    b, n, k, k = local_gram_matrix.shape

    # For each point in the point cloud, compute the difference between the local Gram matrix
    # of the point and the local Gram matrices of all the other points
    repeated_local_gram_matrix = local_gram_matrix.clone().unsqueeze(2).expand(-1, -1, n, -1, -1)
    difference = repeated_local_gram_matrix - local_gram_matrix.unsqueeze(1)

    difference_norm = torch.linalg.matrix_norm(difference, ord="fro")
    # print("difference_norm:", difference_norm.shape)
    # Singular values are returned in descending order
    # print("singular_values.shape:", torch.linalg.svdvals(p_neighborhood).shape)

    # Compute the singular values of each neighborhood matrix which is composed of the
    # features for the point and its neighbors
    singular_values = torch.linalg.svdvals(p_neighborhood)[:,:,-1]
    # print("singular_values.shape:", singular_values.shape)
    # assert torch.sum(singular_values >= 0) == b * n

    # Obtain the minimum singular value
    singular_values = singular_values.unsqueeze(-1)
    # print("singular_values.shape:", singular_values.shape)

    # Calculate the difference between the Frobenius norms and the minimum singular value
    # Points with a negative or 0 difference satisfy the inequality and should be considered as similar points
    frobenius_singular_differences = difference_norm - singular_values/2
    # print("total:",torch.sum(frobenius_singular_differences<=0))

    # Set the original point and k-nearest neighborhood points to have a difference of infinity so they do not
    # also get considered as similar points
    frobenius_singular_differences = frobenius_singular_differences.scatter_(2, index=knn_idx, value=torch.inf)
    # Below is only if you want to filter out point itself from being considered as a similar point
    # frobenius_singular_differences=frobenius_singular_differences.clone().diagonal_scatter(torch.ones(frobenius_singular_differences.shape[:-1])*torch.inf,dim1=-2,dim2=-1)

    smallest_differences, k_smallest_indices=torch.topk(frobenius_singular_differences, k_neighbors, dim=2, largest=False)

    if compute_average_similar:
        average_similar = ((frobenius_singular_differences<=0).sum(dim=-1).sum(dim=-1)/n).sum()/b
    else:
        average_similar = None
    return k_smallest_indices, average_similar

def get_knn_idx(p1, p2, k):
    """Get index of k points of p2 nearest to p1.

    Args:
        p1 (tensor): a batch of point sets with shape of `(b, c, m)`
        p2 (tensor): a batch of point sets with shape of `(b, c, n)`
        k: the number of neighboring points.
    Returns:
        idx (tensor): the index of neighboring points w.r.t p1 in p2
            with shape of `(b, m, k)`.
    """
    # TODO: (10 points) Return the index of the top k elements. Use
    # `torch.topk` and the `pairwise_sqrdist_b` function below.
    #
    # HINT: your intermediate distance array should be of shape (b, m, n) and
    # your index array of shape (b, m, k)
    #

    distances = pairwise_sqrdist_b(p1, p2)

    _, idx = torch.topk(distances, k, dim=2, largest=False)

    return idx


# BELOW are functions provided for your convenience.
def pairwise_sqrdist_b(p, q):
    """Pairwise square distance between two point sets (Batched).

    We implement the memory efficient way to pair-wise distance vis refactorization:
        `(p - q)**2 = p**2 + q**2  - 2*p^T*p`

    Args:
        p (tensor): a batch of point sets with shape of `(b, c, m)`
        q (tensor): a batch of point sets with shape of `(b, c, n)`

    Returns:
        dist (tensor):  pairwise distance matrix.
    """
    dist = -2 * torch.matmul(p.transpose(2, 1), q)  # bmn
    p_sqr = (p ** 2).sum(dim=1, keepdim=True).transpose(2, 1)  # bm1
    q_sqr = (q ** 2).sum(dim=1, keepdim=True)  # b1n
    dist += p_sqr + q_sqr
    return dist


# Modified from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/e365b9f7b9c3d7d6444278d92e298e3f078794e1/models/pointnet2_utils.py#L63
def sampling_fps(xyz, npoint):
    """
    Input:
        xyz (tensor): pointcloud coordinates data with shape of (b, d, n).
        npoint (tensor): number of samples
    Return:
        sampled_pts (tensor): sampled pointcloud index, (b, d, m)
    """
    xyz = xyz.transpose(2, 1)

    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    sampled_pts = index_points(xyz.transpose(2, 1), centroids)
    return sampled_pts



def index_points(points, idx):
    """
    Input:
        points (tensor): input points data of shape (b, c, n),
        idx (tensor): sample index data, [b, s1, s2 ...].
    Return:
        new_points (tensor): indexed points data, [b, c, s1, s2 ..]
    """
    points = points.transpose(2, 1)
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :].moveaxis(-1, 1)
    return new_points



