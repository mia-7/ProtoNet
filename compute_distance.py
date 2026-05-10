import numpy as np
import load_data as loader
import argparse
import torch

# i_d = {
#     0: "Caltech101_7",
#     1: "HandWritten",
#     2: "ALOI_100",
#     3: "YouTubeFace10_4Views",
#     4: "Scene_15",
#     5: "AWA_7",
# }
# parser = argparse.ArgumentParser(description='main_each_epoch')
# parser.add_argument('--i_d', type=int, default='4')
# parser.add_argument("--missrate", default=0.5, type=float)
# args = parser.parse_args()
# i_d = i_d[args.i_d]
# print(i_d)
# my_data_dic = loader.ALL_data
# data_para = my_data_dic[i_d]
# parser.add_argument('--dataset', default=data_para)
# args = parser.parse_args()
# X, Y, missindex, X_com, Y_com, index_com, index_incom = loader.load_data(args.dataset, args.missrate)

def calculate_cluster_distances_wMissing(X, Y, missindex):
    """
    计算每个视角的类内距离和类间距离
    X: 视角特征列表, 每个元素形状为 (N, D_v)
    Y: 真实标签列表 (N,)
    missindex: 掩码矩阵 (N, V), 1表示存在, 0表示缺失
    """
    num_views = len(X)
    view_results = []
    
    # 获取唯一的类别标签
    unique_labels = torch.unique(torch.tensor(Y[0]))
    num_clusters = len(unique_labels)

    for v in range(num_views):
        xv = X[v]
        if isinstance(xv, np.ndarray):
            xv = torch.from_numpy(xv)
        
        # 仅选择在该视角中存在的样本
        mask = missindex[:, v] == 1
        xv_available = xv[mask]
        y_available = torch.tensor(Y[v])[mask]
        
        # 1. 计算每个类的中心 (Centroids)
        centroids = []
        intra_dists = []
        
        for label in unique_labels:
            class_mask = (y_available == label)
            if torch.sum(class_mask) == 0:
                continue
            
            samples_in_class = xv_available[class_mask]
            centroid = torch.mean(samples_in_class, dim=0)
            centroids.append(centroid)
            
            # 计算类内距离 (样本到中心点的平均欧式距离)
            dist_to_centroid = torch.norm(samples_in_class - centroid, dim=1)
            intra_dists.append(torch.mean(dist_to_centroid))
            
        # 2. 计算视角内的类内平均距离
        avg_intra_dist = torch.mean(torch.stack(intra_dists)).item()
        
        # 3. 计算类间距离 (不同中心点之间的平均距离)
        centroids = torch.stack(centroids)
        inter_dist_matrix = torch.cdist(centroids, centroids, p=2)
        
        # 只取上三角部分（避免重复计算和自距离）
        num_c = centroids.size(0)
        triu_indices = torch.triu_indices(num_c, num_c, offset=1)
        avg_inter_dist = torch.mean(inter_dist_matrix[triu_indices[0], triu_indices[1]]).item()
        
        view_results.append({
            'view': v + 1,
            'intra_dist': avg_intra_dist,
            'inter_dist': avg_inter_dist
        })
        
    return view_results

def calculate_cluster_distances(X, Y):
    """
    计算每个视角的类内距离和类间距离
    X: 视角特征列表, 每个元素形状为 (N, D_v)
    Y: 真实标签列表 (N,)
    """
    num_views = len(X)
    view_results = []
    
    # 获取唯一的类别标签
    unique_labels = torch.unique(torch.tensor(Y))
    num_clusters = len(unique_labels)

    for v in range(num_views):
        xv = X[v]
        if isinstance(xv, np.ndarray):
            xv = torch.from_numpy(xv)
        
        xv_available = xv
        y_available = torch.tensor(Y)
        
        # 1. 计算每个类的中心 (Centroids)
        centroids = []
        intra_dists = []
        
        for label in unique_labels:
            class_mask = (y_available == label)
            if torch.sum(class_mask) == 0:
                continue
            
            samples_in_class = xv_available[class_mask]
            centroid = torch.mean(samples_in_class, dim=0)
            centroids.append(centroid)
            
            # 计算类内距离 (样本到中心点的平均欧式距离)
            dist_to_centroid = torch.norm(samples_in_class - centroid, dim=1)
            intra_dists.append(torch.mean(dist_to_centroid))
            
        # 2. 计算视角内的类内平均距离
        avg_intra_dist = torch.mean(torch.stack(intra_dists)).item()
        
        # 3. 计算类间距离 (不同中心点之间的平均距离)
        centroids = torch.stack(centroids)
        inter_dist_matrix = torch.cdist(centroids, centroids, p=2)
        
        # 只取上三角部分（避免重复计算和自距离）
        num_c = centroids.size(0)
        triu_indices = torch.triu_indices(num_c, num_c, offset=1)
        avg_inter_dist = torch.mean(inter_dist_matrix[triu_indices[0], triu_indices[1]]).item()
        
        view_results.append({
            'view': v + 1,
            'intra_dist': avg_intra_dist,
            'inter_dist': avg_inter_dist
        })

        intra_list = [res['intra_dist'] for res in view_results]
        inter_list = [res['inter_dist'] for res in view_results]

    if len(view_results) > 0:
        avg_intra = sum(intra_list) / len(view_results)
        avg_inter = sum(inter_list) / len(view_results)
        print(f"所有视角平均结果：类内距离 = {avg_intra:.4f}, 类间距离 = {avg_inter:.4f}")
        
    return view_results

# # --- 调用示例 ---
# # 假设 X, Y, missindex 已通过 loader.load_data 获取
# results = calculate_cluster_distances_wMissing(X, Y, missindex)

# for res in results:
#     print(f"视角 {res['view']}: 类内距离 = {res['intra_dist']:.4f}, 类间距离 = {res['inter_dist']:.4f}")

# intra_list = [res['intra_dist'] for res in results]
# inter_list = [res['inter_dist'] for res in results]

# if len(results) > 0:
#     avg_intra = sum(intra_list) / len(results)
#     avg_inter = sum(inter_list) / len(results)
#     print(f"所有视角平均结果：类内距离 = {avg_intra:.4f}, 类间距离 = {avg_inter:.4f}")