import os
import time
from compute_distance import calculate_cluster_distances
import torch
from Nmetrics import evaluate
import numpy as np
from tqdm import tqdm
import argparse
import datetime
from pathlib import Path
import random
import load_data as loader
from prior.network import Network
from loss import Cross_inscl_loss, Noise_robust_loss
from datasets import Data_Sampler, TrainDataset_Com, TrainDataset_All
from sklearn.cluster import KMeans
import utils
from utils import get_Similarity, euclidean_dist
import matplotlib
import scipy.io as sio
matplotlib.use('Agg')


def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


def pretrain(model, opt_pre, args, device, X_com, Y_com, X, Y):
    train_dataset = TrainDataset_Com(X_com, Y_com)
    batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    t_progress = tqdm(range(args.pretrain_epochs), desc='Pretraining')

    for epoch in t_progress:
        tot_loss = 0.0
        loss_fn = torch.nn.MSELoss()
        for batch_idx, (xs, ys) in enumerate(train_loader):
            for v in range(args.V):
                xs[v] = torch.squeeze(xs[v]).to(device)

            opt_pre.zero_grad()
            zs, xrs = model(xs)
            loss_list = []
            for v in range(args.V):
                loss_value = loss_fn(xs[v], xrs[v])
                loss_list.append(loss_value)
            loss = sum(loss_list)
            loss.backward()
            opt_pre.step()
            tot_loss += loss.item()
        # print('Epoch {}'.format(epoch + 1), 'Loss:{:.6f}'.format(tot_loss / len(train_loader)))

    fea_emb = [[] for _ in range(args.V)]

    all_dataset = TrainDataset_Com(X, Y)
    batch_sampler_all = Data_Sampler(all_dataset, shuffle=False, batch_size=args.batch_size, drop_last=False)
    all_loader = torch.utils.data.DataLoader(dataset=all_dataset,
                                             batch_sampler=batch_sampler_all)
    with torch.no_grad():
        for batch_idx2, (xs2, _) in enumerate(all_loader):
            for v in range(args.V):
                xs2[v] = torch.squeeze(xs2[v]).to(device)
            zs2, xrs2 = model(xs2)
            for v in range(args.V):
                zs2[v] = zs2[v].cpu()
                fea_emb[v] = fea_emb[v] + zs2[v].tolist()

    for v in range(args.V):
        fea_emb[v] = torch.tensor(fea_emb[v])

    return fea_emb


def train_align(state_logger, decoder_model, opt_align, args, device, X, Y, Miss_vecs, proto_Num, missindex,
                final_batch, r, ProtoRobs_epochs, F_ind):
    train_dataset = TrainDataset_All(X, Y, Miss_vecs)
    # current_x_list, current_y_list, current_miss_list
    batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=args.Batch_Rob, drop_last=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    time0 = time.time()
    t_progress = tqdm(range(ProtoRobs_epochs), desc='RobustTraining')
    acc_best, nmi_best, fscore_best, epoch_best = 0., 0., 0., 0
    state_logger.write('Start time: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    state_logger.write('\n>> Start training {}-th initial, seed: {},'.format(args.train_id, args.seed))
    loss_his, metric_his = [[], [], [], [], []], [[], [], []]

    for epoch in t_progress:
        loss_recon_epoch, loss_ins_cl_epoch, loss_Rob_epoch, loss_tnn_epoch, loss_total_epoch = 0., 0., 0., 0., 0
        AllPrototypes = [torch.tensor([]) for _ in range(args.V)]
        for batch_idx, (x, y, miss_vec) in enumerate(train_loader):
            opt_align.zero_grad()
            loss_fn = torch.nn.MSELoss().to(device)
            proto_Noiserobust = Noise_robust_loss().to(device)
            ins_contra = Cross_inscl_loss().to(device)

            loss_list_recon = []
            loss_list_ins = []
            loss_list_Rob = []
            Prototypes = [[] for _ in range(args.V)]

            for v in range(args.V):
                x[v] = torch.squeeze(x[v]).to(device)
                y[v] = torch.squeeze(y[v]).to(device)
                miss_vec[v] = torch.squeeze(miss_vec[v]).to(device)

            z, xr = decoder_model(x)
            for v in range(args.V):
                loss_list_recon.append(loss_fn(x[v][miss_vec[v] > 0], xr[v][miss_vec[v] > 0]))

            loss_recon = sum(loss_list_recon)

            for v1 in range(args.V):
                v2_start = v1 + 1
                for v2 in range(v2_start, args.V):
                    align_index = []
                    for i in range(x[0].shape[0]):
                        if miss_vec[v1][i] == 1 and miss_vec[v2][i] == 1:
                            align_index.append(i)

                    z1 = z[v1][align_index]
                    z2 = z[v2][align_index]
                    l_inscontra = ins_contra(z1, z2)
                    loss_list_ins.append(l_inscontra)

            loss_ins_cl = sum(loss_list_ins)

            for v1 in range(args.V):
                align_index = []
                for i in range(x[0].shape[0]):
                    if miss_vec[v1][i] == 1:
                        align_index.append(i)
                Feature = z[v1][align_index]
                Pk = proto_Num[batch_idx]
                Pk = int(Pk)
                F = Feature[:, v1]
                size = F.size()[0]
                if Pk > size:
                    Pk = size
                Prototypes[v1] = decoder_model.updateC(Feature, Pk, device)
                # AllPrototypes[v1] = AllPrototypes[v1].to(device).detach()
                # AllPrototypes[v1] = torch.cat((AllPrototypes[v1], Prototypes[v1]), dim=0)
                initial_prototypes = Prototypes[v1].detach()
                AllPrototypes[v1] = AllPrototypes[v1].to(device)
                AllPrototypes[v1] = torch.cat((AllPrototypes[v1], initial_prototypes), dim=0)
                """Prototypes[v1]: prototypes for current batch
                AllPrototypes[v1]: prototypes for all batches at one epoch"""
            # for v in range(args.V):
            #     AllPrototypes[v] = torch.tensor(AllPrototypes[v]).to(device)

            # This step exists error, z need be revised
            z_Incomplete = []
            for v1 in range(args.V):
                z_Incomplete.append(z[v1] * miss_vec[v1].unsqueeze(1).float())
            z_En, s = decoder_model.imputeZ(z_Incomplete, Prototypes, device) #

            '''Imputation'''
            st = torch.stack(s) # V*N*P
            mask = torch.zeros_like(st)
            for i in range(st.shape[1]):
                max_values = st[:, i, :].max(dim=1).values
                is_zero_row = (st[:, i, :].sum(dim=1) == 0) 
                mask_row = (st[:, i, :] == max_values.unsqueeze(1)).float()
                mask_row[is_zero_row] = 0  
                mask[:, i, :] = mask_row
                # mask[:, i, :] = (st[:, i, :] == max_values.unsqueeze(1)).float()
            st_ = st * mask

            Protot = torch.stack(Prototypes)  # V*P*d
            z_Imp = torch.matmul(st_, Protot)
            for v in range(args.V):
                for i in range (st.shape[1]):
                    if miss_vec[v][i] == 0:
                        z_Imp[v, i, :] = z_Imp[:, i, :].sum(dim=0) / sum(row[i] for row in miss_vec)

            '''Add Z_Imp'''
            s_tensor = torch.stack(z_En) + z_Imp 

            s_hat = torch.fft.fft(s_tensor, dim=1)
            loss_tnn = 0
            for i in range(s_hat.shape[1]):
                _, value, _ = torch.linalg.svd(s_hat[:, i, :])
                loss_tnn += value.sum()

            s_hat = torch.fft.fft(s_tensor, dim=-1)
            for i in range(s_hat.shape[-1]):
                _, value, _ = torch.linalg.svd(s_hat[:, :, i])
                loss_tnn += value.sum()

            for v1 in range(args.V):
                v2_start = v1 + 1
                for v2 in range(v2_start, args.V):
                    prov1 = Prototypes[v1]
                    prov2 = Prototypes[v2]
                    l_Rob = proto_Noiserobust(prov1, prov2, r)
                    loss_list_Rob.append(l_Rob)
            loss_Rob = sum(loss_list_Rob)

            """total_loss"""
            loss_total = loss_recon + args.para_loss[0] * loss_ins_cl + args.para_loss[1] * loss_Rob + args.para_loss[
                2] * loss_tnn  #
            loss_total.backward()
            opt_align.step()

            """record loss"""
            loss_recon_epoch += loss_recon.item()
            loss_ins_cl_epoch += args.para_loss[0] * loss_ins_cl.item()
            loss_Rob_epoch += args.para_loss[1] * loss_Rob.item()
            loss_tnn_epoch += args.para_loss[2] * loss_tnn.item()
            loss_total_epoch += loss_total.item()
        if epoch % args.print_num ==0:
            fea_all = []
            for v in range(args.V):
                fea_all.append([])

            all_dataset = TrainDataset_Com(X, Y)
            batch_sampler_all = Data_Sampler(all_dataset, shuffle=False, batch_size=args.batch_size, drop_last=False)
            all_loader = torch.utils.data.DataLoader(dataset=all_dataset, batch_sampler=batch_sampler_all)
            with torch.no_grad():
                for batch_idx2, (xs2, _) in enumerate(all_loader):
                    for v in range(args.V):  #
                        xs2[v] = torch.squeeze(xs2[v]).to(device)
                    zs2, xrs2 = decoder_model(xs2)
                    for v in range(args.V):
                        zs2[v] = zs2[v].cpu()
                        fea_all[v] = fea_all[v] + zs2[v].tolist()

            for v in range(args.V):
                fea_all[v] = torch.tensor(fea_all[v])

            Proto_Align = []
            for v in range(args.V):
                Proto_Align.append([])
            for v1 in range(args.V):
                v2_start = v1 + 1
                for v2 in range(v2_start, args.V):
                    prov1 = AllPrototypes[v1]
                    prov2 = AllPrototypes[v2]
                    C = euclidean_dist(prov1, prov2)
                    aligen_num = len(prov1)
                    align_out0 = []
                    align_out1 = []
                    for i in range(aligen_num):
                        idx = torch.argsort(C[i, :])
                        align_out0.append((prov1[i, :].detach().cpu()).numpy())
                        align_out1.append((prov2[idx[0], :].detach().cpu()).numpy())
                    Proto_Align[v1], Proto_Align[v2] = torch.from_numpy(np.array(align_out0)).to(device), torch.from_numpy(
                        np.array(align_out1)).to(device)
            epoch_time = time.time() - time0

            all_dataset2 = TrainDataset_Com(fea_all, Y)
            batch_sampler_all2 = Data_Sampler(all_dataset2, shuffle=False, batch_size=final_batch, drop_last=False)
            all_loader2 = torch.utils.data.DataLoader(dataset=all_dataset2, batch_sampler=batch_sampler_all2)

            fea_final = []
            for v in range(args.V):
                fea_final.append([])

            for batch_idx, (xs, ys) in enumerate(all_loader2):
                for v in range(args.V):
                    xs[v] = torch.squeeze(xs[v]).to(device)
                    Proto_Align[v] = torch.squeeze(Proto_Align[v]).to(device)
                
                cossim_mat = []
                for v in range(args.V):
                    sim_mat = get_Similarity(Proto_Align[v], xs[v])
                    sim_mat1 = get_Similarity(xs[v], xs[v])
                    diag = torch.diag(sim_mat1)
                    sim_diag = torch.diag_embed(diag)
                    sim_mat1 = sim_mat1 - sim_diag
                    for i in range(xs[0].shape[0]):
                        if missindex[final_batch * batch_idx + i, v] == 0:
                            sim_mat1[:, i] = 0
                            sim_mat1[i, :] = 0
                    cossim_mat.append(sim_mat1)
                # the size of cossim_mat is N*Pk

                for i in range(xs[0].shape[0]):
                    for v in range(args.V):
                        if missindex[final_batch * batch_idx + i, v] == 0:
                            bc = 0
                            a = 0
                            for v1 in [t for t in range(args.V) if t != v]:
                                if missindex[final_batch * batch_idx + i, v1] == 1:
                                    vec_tmp = cossim_mat[v1][i]
                                    _, indices = torch.sort(vec_tmp, descending=True)
                                    bc = bc + xs[v][indices[0]]
                                    a = a + 1
                            bc = bc / a
                            xs[v][i] = bc
                
                cossim_mat = []
                for v in range(args.V):
                    sim_mat = get_Similarity(Proto_Align[v], xs[v])
                    cossim_mat.append(sim_mat.t())

                for i in range(xs[0].shape[0]):
                    imfu = []
                    for v in range(args.V):
                        imfu.append([])
                    bc = 0
                    a = 0
                    for v in range(args.V):
                        if missindex[final_batch * batch_idx + i, v] == 0:
                            vec_tmp = cossim_mat[v][i]  # Pk
                            _, indices = torch.sort(vec_tmp, descending=True)
                            for v in range(args.V):
                                imfu[v] = Proto_Align[v][indices[0]]
                                bc = bc + Proto_Align[v][indices[0]]
                                a = a + 1
                            bc = bc / a
                            xs[v][i] = bc

                for v in range(args.V):
                    fea_final[v] = fea_final[v] + xs[v].tolist()
            for v in range(args.V):
                fea_final[v] = torch.tensor(fea_final[v])

            for v in range(args.V):
                fea_final[v] = fea_final[v].cpu()
            Labels = Y[0]
            estimator = KMeans(n_clusters=args.K)
            fea_cluster = fea_final[0]
            for i in range(1, len(fea_final)):
                fea_cluster = np.concatenate((fea_cluster, fea_final[i]), axis=1)
            
            ''' calculating distances '''
            # from compute_distance import calculate_cluster_distances
            # calculate_cluster_distances(fea_final, Labels)
 
            estimator.fit(fea_cluster)
            pred_final = estimator.labels_
            acc, nmi, purity, fscore, precision, recall, ari = evaluate(Labels, pred_final)

            state_logger.write('Epoch {}, Loss = {}, K-means: ACC = {:.2f} NMI = {:.2f} fscore = {:.2f}'
                            .format(epoch, loss_total_epoch, acc * 100, nmi * 100, fscore * 100))

            loss_his[0].append(loss_total_epoch)
            loss_his[1].append(loss_recon_epoch)
            loss_his[2].append(loss_ins_cl_epoch)
            loss_his[3].append(loss_Rob_epoch)
            loss_his[4].append(loss_tnn_epoch)
            metric_his[0].append(acc)
            metric_his[1].append(nmi)
            metric_his[2].append(fscore)
            # utils.plot_loss_and_metric(loss_his, metric_his, filename=os.path.join(args.image_dir, str(args.seed)),
            #                         loss_name=['total', 'recon', 'ins_cl', 'Rob', 'tnn_zImp'],
            #                         metric_name=['ACC', 'NMI', 'F'])

            print('Epoch=', epoch, 'acc=', acc * 100, 'nmi=', nmi * 100, 'fscore=', fscore * 100, 'ari=', ari * 100,
                'recall=', recall * 100,
                'precision=', precision * 100)

            if acc > acc_best:
                acc_best = acc
                nmi_best = nmi
                fscore_best = fscore
                epoch_time = epoch

                # ================= Inter-class and Intra-class distance computing =================
                print(f"\n---> [Epoch {epoch}] computing distance of imputed feature ...")
                calculate_cluster_distances(fea_final, Labels)
                dataset_name = args.dataset[1]
                miss_rate = args.missrate
                fea_final_np = [f.detach().cpu().numpy() if torch.is_tensor(f) else np.array(f) for f in fea_final]
                labels_np = Labels.cpu().numpy() if torch.is_tensor(Labels) else np.array(Labels)
                mat_filename = f"{dataset_name}_mr{miss_rate}_fea_final.mat"
                sio.savemat(mat_filename, {'fea_final': fea_final_np, 'Labels': labels_np})
                print(f"---> Imputation is saved to : {mat_filename}")
                # for v in range(args.V):
                #     fea_final[v] = fea_all[v]
    print('Best_Epoch=', epoch_time, 'acc=', acc_best * 100, 'nmi=', nmi_best * 100)
    state_logger.write('Best_Epoch {} K-means: ACC = {:.2f} NMI = {:.2f} Fscore = {:.2f}'
                       .format(epoch_time, acc_best * 100, nmi_best * 100, fscore_best * 100))

    train_state = {'acc': acc_best, 'nmi': nmi_best, 'fscore': fscore_best}
    return train_state


"""  python main.py --i_d 0 --missrate 0.3 """
i_d = {
    0: "Caltech101_7",
    1: "HandWritten",
    2: "ALOI_100",
    3: "YouTubeFace10_4Views",
    4: "Scene_15",
    5: "AWA_7",
}
parser = argparse.ArgumentParser(description='main_each_epoch')
parser.add_argument('--i_d', type=int, default='4')
parser.add_argument('--device', type=str, default='cuda:7')
parser.add_argument("--protorate", default=0.3, type=float)
parser.add_argument("--r", default=0.5, type=float)
parser.add_argument("--missrate", default=0.5, type=float)
parser.add_argument('--train_time', type=int, default=1)
parser.add_argument('--print_num', type=int, default=1)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--lr_align', default=0.0001, type=float)
parser.add_argument('--output_dir', type=str, default='./ProtoIMC/tnn_wImpute_wCL1_repeat/',
                    help='path where to save, empty for no saving')
parser.add_argument('--para_loss', type=float, nargs=3, help="List of loss parameters", required=True)
args = parser.parse_args()
i_d = i_d[args.i_d]
print(i_d)
my_data_dic = loader.ALL_data
data_para = my_data_dic[i_d]
parser.add_argument('--dataset', default=data_para)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--Batch_Rob', default=256, type=int)
parser.add_argument('--lr_pre', default=0.0005, type=float)
parser.add_argument('--pretrain_epochs', default=200, type=int)
parser.add_argument('--ProtoRobs_epochs', default=100, type=int)
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--final_batch", default=256)
parser.add_argument("--V", default=data_para['V'])
parser.add_argument("--K", default=data_para['K'])
parser.add_argument("--N", default=data_para['N'])
parser.add_argument("--view_dims", default=data_para['n_input'])
# parser.add_argument('--para_loss', default=data_para['para_loss'])
args = parser.parse_args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(args.device)


def main(): 
    for t in range(1): #args.train_time
        args.seed = args.train_time
        seed_everything(args.seed)
        X, Y, missindex, X_com, Y_com, index_com, index_incom = loader.load_data(args.dataset, args.missrate)
        Miss_vecs = []
        for v in range(args.V):
            Miss_vecs.append(missindex[:, v])
        decoder_model = Network(args.V, args.view_dims, args.feature_dim).to(device)

        Protonum = len(X_com[0])
        n = args.N // args.Batch_Rob

        Num_C = (Protonum * args.protorate)
        c_num = Num_C // n  # the number of prototype within each batch
        proto_Num = [[] for _ in range(n)]
        for i in range(n):
            proto_Num[i] = c_num  # // args.K * args.K
        # numclass = len(np.unique(Y))
        # anchor = int(proto_Num[0] / numclass)
        # similar_value = 1 / anchor
        # from scipy.linalg import toeplitz
        # a = toeplitz([similar_value] * anchor)
        # b = [a] * numclass
        # F_ind = np.block([[b[i] if i == j else np.zeros_like(a) for j in range(numclass)] for i in range(numclass)])
        F_ind = np.zeros(1)

        print('+' * 30, ' Parameters ', '+' * 30)
        print(args)
        print('+' * 75)

        out = './pre_models/' + args.dataset[1] + '_{}.pth'.format(args.missrate)
        if True:#os.path.exists(out):
            checkpoint = torch.load(out, map_location='cpu')
            decoder_model.load_state_dict(checkpoint['net'])
        else:
            optimizer_pretrain = torch.optim.Adam(decoder_model.parameters(), lr=args.lr_pre)
            _ = pretrain(decoder_model, optimizer_pretrain, args, device, X_com, Y_com, X, Y)
            state = {'net': decoder_model.state_dict()}
            torch.save(state, out)
            args.reload = True

        start_time = time.time()
        result_avr = {'acc': [], 'nmi': [], 'fscore': []}
        state_logger = utils.FileLogger(os.path.join(args.output_dir, 'log_train.txt'))

        state_logger.write('=========Prototype rate {:.2f}========='
                       .format(args.protorate))
        args.train_id = t
        optimizer_align = torch.optim.Adam(decoder_model.parameters(), lr=args.lr_align)
        train_state = train_align(state_logger, decoder_model, optimizer_align, args, device, X, Y, Miss_vecs,
                                  proto_Num,
                                  missindex, args.final_batch, args.r, args.ProtoRobs_epochs, F_ind)
        
        # args.seed = args.seed + 1
        # args.seed = (args.seed + datetime.datetime.now().microsecond) % 999

        for k, v in train_state.items():
            result_avr[k].append(v)

    for k, v in result_avr.items():
        x = np.asarray(v)
        result_avr[k] = [x.mean(), x.std()]

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    state_logger.write('\nTraining time {}\n'.format(total_time_str))
    state_logger.write('Average K-means Result: ACC = {:.4f}({:.4f}) NMI = {:.4f}({:.4f}) Fscore = {:.4f}({:.4f})'
                       .format(*result_avr['acc'], *result_avr['nmi'], *result_avr['fscore']))
    return result_avr['acc'], result_avr['nmi'], result_avr['fscore']


if __name__ == '__main__':
    folder_name = '_'.join(
        [args.dataset[1], 'msrt', str(args.missrate)])
    args.output_dir = os.path.join(args.output_dir, folder_name)

    if args.para_loss[2] == 0.:
        subfolder_name = '_'.join([str(args.para_loss[2])])
        args.output_dir = os.path.join(args.output_dir, subfolder_name)

        subfolder_name = '_'.join([str(args.para_loss[0]), str(args.para_loss[1])])
        args.output_dir = os.path.join(args.output_dir, subfolder_name)
    else:
        subfolder_name = '_'.join([str(args.para_loss[0]), str(args.para_loss[1])])
        args.output_dir = os.path.join(args.output_dir, subfolder_name)

        subfolder_name = '_'.join([str(args.para_loss[2]), 'Grad_FT'])
        args.output_dir = os.path.join(args.output_dir, subfolder_name)

    args.image_dir = args.output_dir  # os.path.join(args.output_dir, 'visualize')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Path(os.path.join(args.output_dir, 'visualize')).mkdir(parents=True, exist_ok=True)
    acc, nmi, fscore = main()
    acc_ = acc[0]
    nmi_ = nmi[0]
    fscore_ = fscore[0]
    print(f"{acc_:.4f},{nmi_:.4f},{fscore_:.4f}") 
