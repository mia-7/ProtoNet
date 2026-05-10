import torch
import numpy as np
import warnings
import random
import torch.nn.functional as F
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from kmeans_gpu import kmeans
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def NormalizeFeaTorch(features):
    rowsum = torch.tensor((features ** 2).sum(1))
    r_inv = torch.pow(rowsum, -0.5)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    normalized_feas = torch.mm(r_mat_inv, features)
    return normalized_feas



def get_Similarity(fea_mat1, fea_mat2):
    Sim_mat = F.cosine_similarity(fea_mat1.unsqueeze(1), fea_mat2.unsqueeze(0), dim=-1)

    return Sim_mat


def clustering(feature, cluster_num):

    predict_labels,  cluster_centers = kmeans(X=feature, num_clusters=cluster_num, distance='euclidean', device=device)
    return predict_labels.numpy(), cluster_centers




def euclidean_dist(x, y, root=False):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    if root:
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def tensor_nuclear_norm(X):
    X_hat = torch.fft.fft(X, dim=-1)  # FFT 沿着第 3 维
    tnn = 0
    for i in range(X.shape[2]):  # 遍历每个 frontal slice
        _, S, _ = torch.linalg.svd(X_hat[:, :, i])  # SVD
        tnn += S.sum()  # 计算核范数（奇异值之和）
    return tnn / X.shape[2]  # 归一化

class FileLogger:
    def __init__(self, output_file):
        self.output_file = output_file

    def write(self, msg, p=True):
        with open(self.output_file, mode="a", encoding="utf-8") as log_file:
            log_file.writelines(msg + '\n')
        if p:
            print(msg)


def plot_loss_and_metric(loss_history, metric_history, filename='loss_metric_plot.png', loss_name=[], metric_name=[]):
    """
    """
    num_loss = len(loss_name)
    epochs = range(1, len(loss_history[-1]) + 1)

    fig, axes = plt.subplots(num_loss+1, 1, figsize=(10, 4*(num_loss+1)))

    for i in range(num_loss):
        axes[i].plot(epochs, loss_history[i], label=loss_name[i])
        axes[i].set_title(f'{loss_name[i]} vs Epoch')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(loss_name[i])
        axes[i].legend()

    for i in range(len(metric_name)):
        axes[-1].plot(epochs, metric_history[i], label=metric_name[i])
    axes[-1].set_title('Metric vs Epoch')
    axes[-1].set_xlabel('Epoch')
    axes[-1].set_ylabel('Metric')
    axes[-1].legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def pairwise_js_divergence(P, Q, eps=1e-8):
    """
    Compute the JSD between P_i and Q_i
    Parameters:
        P (Tensor): [N, K]
        Q (Tensor): [N, K]
        eps (float): avoid log(0)
    Returns:
        Tensor: [N,] JSD Matrix
    """
    P = P + eps
    Q = Q + eps
    P = P / P.sum(dim=1, keepdim=True)
    Q = Q / Q.sum(dim=1, keepdim=True)

    logP = P.log()
    logQ = Q.log()

    M = 0.5 * (P + Q)  # [N, K]
    logM = M.log()

    kl_PM = (P * (logP - logM)).sum(dim=-1)  # [N,]
    kl_QM = (Q * (logQ - logM)).sum(dim=-1)  # [N,]
    jsd = 0.5 * (kl_PM + kl_QM)

    return jsd.sum()

def entropy_of_matrix(C: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    C_clamped = C.clamp(min=eps, max=1 - eps)
    entropy = - (C_clamped * torch.log(C_clamped) +
                 (1 - C_clamped) * torch.log(1 - C_clamped)).sum()
    return entropy

import torch

def kmeans_gpu(X, k, max_iters=300, tol=1e-4, init='k-means++', device='cuda'):
    """
    Perform K-means clustering on GPU using PyTorch, similar to sklearn's KMeans.
    
    Args:
        X (torch.Tensor): The input data (n_samples, n_features).
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for convergence (centroid shift).
        init (str): Initialization method ('random' or 'k-means++').
        device (str): The device to run the computations on ('cuda' or 'cpu').
    
    Returns:
        centroids (torch.Tensor): Final centroids of the clusters.
        labels (torch.Tensor): Labels of each data point.
    """
    # Move data to GPU
    X = X.to(device)
    
    # Initialize centroids
    if init == 'k-means++':
        centroids = kmeans_plus_plus(X, k, device)
    elif init == 'random':
        centroids = X[torch.randint(0, X.size(0), (k,))].clone()
    else:
        raise ValueError("Unknown initialization method")
    
    # Initialize variables
    prev_centroids = centroids.clone()
    labels = torch.zeros(X.size(0), dtype=torch.long, device=device)
    
    for _ in range(max_iters):
        # Step 1: Compute distances from each point to each centroid
        dist = torch.cdist(X, centroids, p=2)  # Euclidean distance
        new_labels = torch.argmin(dist, dim=1)  # Assign points to the closest centroid
        
        # Step 2: Check for convergence (if the labels don't change)
        if torch.all(new_labels == labels):
            print(f"Converged at iteration {_}")
            break
        labels = new_labels
        
        # Step 3: Update centroids by computing the mean of points in each cluster
        for i in range(k):
            centroids[i] = X[labels == i].mean(dim=0)
        
        # Step 4: Check if centroids have changed significantly
        centroid_shift = torch.norm(centroids - prev_centroids).item()
        if centroid_shift < tol:
            print(f"Centroids converged at iteration {_}")
            break
        prev_centroids = centroids.clone()
    
    return centroids, labels

def kmeans_plus_plus(X, k, device):
    """
    Initialize centroids using the k-means++ method.
    
    Args:
        X (torch.Tensor): The input data (n_samples, n_features).
        k (int): Number of clusters.
        device (str): The device to run the computations on ('cuda' or 'cpu').
    
    Returns:
        centroids (torch.Tensor): Initialized centroids.
    """
    n_samples, n_features = X.size()
    
    # Initialize the first centroid randomly
    centroids = X[torch.randint(0, n_samples, (1,))].clone()
    
    for _ in range(1, k):
        # Compute the distance of each point to the closest centroid
        dist = torch.cdist(X, centroids, p=2)  # Euclidean distance
        min_dist = torch.min(dist, dim=1)[0]  # Distance to the nearest centroid
        
        # Choose the next centroid with probability proportional to the squared distance
        prob = min_dist**2
        prob = prob / prob.sum()  # Normalize the probabilities
        new_centroid_idx = torch.multinomial(prob, 1)
        new_centroid = X[new_centroid_idx].clone()
        
        # Add the new centroid to the list of centroids
        centroids = torch.cat([centroids, new_centroid], dim=0)
    
    return centroids




