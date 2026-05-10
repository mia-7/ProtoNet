import os

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import h5py
import random
import warnings
warnings.filterwarnings("ignore")


# 需要h5py读取
ALL_data = dict(
    Caltech101_7= {1: 'Caltech101_7', 'N': 1400, 'K': 7, 'V': 5, 'n_input': [1984, 512, 928, 254, 40],'para_loss': [1e-4, 1e-2]},
    HandWritten = {1: 'handwritten1031_v73', 'N': 2000, 'K': 10, 'V': 6, 'n_input': [240, 76, 216, 47, 64, 6],'para_loss': [1e-4, 1e-2]},
    ALOI_100 = {1: 'ALOI_100_7', 'N': 10800, 'K': 100, 'V': 4, 'n_input': [77, 13, 64, 125],'para_loss': [1e-3, 1e-3]},
    YouTubeFace10_4Views={1: 'YTF10_4', 'N': 38654, 'K': 10, 'V': 4, 'n_input': [944, 576, 512, 640],'para_loss': [1e-4, 1e-2]},
    Scene_15 = {1: 'Scene_15_v73', 'N': 4485, 'K': 15, 'V': 3, 'n_input': [20, 59, 40], 'para_loss': [1e-4, 1e-2, 1e-2]},
    AWA_7 ={1: 'AWA_73', 'N': 10158, 'K': 50, 'V': 7, 'n_input': [2688, 2000, 2000, 2000, 2000, 4096, 4096],'para_loss': [1e-3, 1e-3]}
)


path = 'Datasets/'


def get_mask(view_num, alldata_len, missing_rate):

    missindex = np.ones((alldata_len, view_num))
    b=((10 - 10*missing_rate)/10) * alldata_len
    miss_begin = int(b)
    for i in range(miss_begin, alldata_len):
        missdata = np.random.randint(0, high=view_num,
                                     size=view_num - 1) # larger than or equal to 0, less than view_num

        missindex[i, missdata] = 0

    return missindex

def get_adversarial_mask(X_list, y, missing_rate, mode='feature_dependent'):
    """
    生成非随机 (Non-random) 和 视角相关 (View-dependent) 的缺失掩码
    
    参数:
    X_list: list of np.array, 原始多视图数据 [[N, d1], [N, d2], ...]
    y: np.array, 标签 [N] (用于 class_dependent 模式)
    missing_rate: float, 目标总体缺失率
    mode: str, 'feature_dependent' (基于特征强度的对抗缺失) 
               'class_dependent' (基于类别的结构化缺失)
    """
    num_views = len(X_list)
    N = X_list[0].shape[0]
    mask = np.ones((N, num_views))
    
    # 计算需要缺失的总单元格数 (不含全部缺失的情况)
    total_entries = N * num_views
    num_to_miss = int(total_entries * missing_rate)
    
    if mode == 'feature_dependent':
        # 模拟“对抗性”：特征强度越高（或信噪比极端）的样本越容易缺失
        # 这在实际中模拟传感器量程溢出或环境干扰导致的失效
        probs = []
        for v in range(num_views):
            # 计算每个样本在该视角下的特征范数
            norm = np.linalg.norm(X_list[v], axis=1)
            # 范数越大，缺失概率越高 (线性映射)
            p = norm / np.sum(norm)
            probs.append(p)
        
        # 展平概率并进行采样
        flat_probs = np.array(probs).T.flatten() # [N * num_views]
        indices = np.random.choice(total_entries, size=num_to_miss, replace=False, p=flat_probs/flat_probs.sum())
        
    elif mode == 'class_dependent':
        # 模拟“视角相关”：特定类别的样本在某些视角下天然缺失严重
        # 例如：类别 1 在视角 0 缺失率极高，类别 2 在视角 1 缺失率极高
        flat_probs = np.zeros((N, num_views))
        unique_classes = np.unique(y)
        
        for c in unique_classes:
            class_idx = (y == c)
            # 为每个类别分配一个“脆弱视角”
            v_vulnerable = int(c % num_views)
            flat_probs[class_idx, v_vulnerable] = 0.8 # 该视角 80% 概率权重
            # 其他视角平分剩余权重
            other_views = [v for v in range(num_views) if v != v_vulnerable]
            flat_probs[class_idx, np.array(other_views)] = 0.2 / (num_views - 1)
            
        flat_probs = flat_probs.flatten()
        indices = np.random.choice(total_entries, size=num_to_miss, replace=False, p=flat_probs/flat_probs.sum())

    # 应用掩码
    rows, cols = np.unravel_index(indices, (N, num_views))
    mask[rows, cols] = 0
    
    # 安全检查：确保每个样本至少保留一个视角
    for i in range(N):
        if np.sum(mask[i, :]) == 0:
            keep_v = np.random.randint(0, num_views)
            mask[i, keep_v] = 1
            
    return mask


def Form_Incomplete_Data(missrate=0.5, X = [], Y = []):
    # X, Y: ground truth
    # missindex: Mask matrix with size N*V
    # filled_X_complete, filled_Y_complete: Complete data with the same size from all views
    # index_complete, index_partial: size V*Lv, V*(L-Lv)

    np.random.seed(1)

    size = len(Y[0])
    view_num = len(X)
    index = [i for i in range(size)]
    np.random.shuffle(index)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]

    missindex = get_mask(view_num, size, missrate) # missindex_ij = 0 indicates that i-th sample misses in the j-th view.

    index_complete = []
    index_partial = []
    for i in range(view_num):
        index_complete.append([])
        index_partial.append([])
    for i in range(missindex.shape[0]):
        for j in range(view_num):
            if missindex[i, j] == 1:
                index_complete[j].append(i)
            else:
                index_partial[j].append(i)
    # Max_len: the maximum length of the complete data from different views
    # filled_index_com: fill the index from the complete index to reach the length of Max_len
    filled_index_com = []
    for i in range(view_num):
        filled_index_com.append([])
    max_len = 0
    for v in range(view_num):
        if max_len < len(index_complete[v]):
            max_len = len(index_complete[v])
    for v in range(view_num):
        if len(index_complete[v]) < max_len:
            diff_len = max_len - len(index_complete[v])

            diff_value = random.sample(index_complete[v], diff_len)
            filled_index_com[v] = index_complete[v] + diff_value
        elif len(index_complete[v]) == max_len:
            filled_index_com[v] = index_complete[v]

    filled_X_complete = []
    filled_Y_complete = []
    for i in range(view_num):
        filled_X_complete.append([])
        filled_Y_complete.append([])
        filled_X_complete[i] = X[i][filled_index_com[i]]
        filled_Y_complete[i] = Y[i][filled_index_com[i]]

    for v in range(view_num):
        X[v] = torch.from_numpy(X[v])
        filled_X_complete[v] = torch.from_numpy(filled_X_complete[v])

    return X, Y, missindex, filled_X_complete, filled_Y_complete, index_complete, index_partial

def load_data(dataset, missrate):

    data = h5py.File(path + dataset[1] + ".mat")
    X = []
    Y = []
    Label = np.array(data['Y']).T

    Label = Label.reshape(Label.shape[0])
    mm = MinMaxScaler()
    for i in range(data['X'].shape[1]):
        diff_view = data[data['X'][0, i]]
        diff_view = np.array(diff_view, dtype=np.float32).T
        std_view = mm.fit_transform(diff_view)
        X.append(std_view)
        Y.append(Label)
    X, Y, missindex, X_com, Y_com, index_com, index_incom = Form_Incomplete_Data(missrate=missrate, X=X, Y=Y)

    return X, Y, missindex, X_com, Y_com, index_com, index_incom



