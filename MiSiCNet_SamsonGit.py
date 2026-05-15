# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 19:13:50 2021

@author: behnood
"""

#from __future__ import print_function
import matplotlib.pyplot as plt
#%matplotlib inline
# from numpy import linalg as LA
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 取得目前這支 Python 程式所在的資料夾，之後所有資料路徑都從這裡開始找
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
#from models import *
#import math
import torch
import torch.optim
import torch.nn as nn

# from skimage.measure import compare_psnr
# from skimage.measure import compare_mse
#from utils.denoising_utils import *

# from skimage._shared import *
# from skimage.util import *
# from skimage.metrics.simple_metrics import _as_floats
# from skimage.metrics.simple_metrics import mean_squared_error

#from UtilityMine import add_noise
# from UtilityMine import find_endmember
# from UtilityMine import add_noise
from UtilityMine import *
# from VCA import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

PLOT = False  # 不顯示訓練過程圖片
PRINT_PROGRESS = False  # True 才會每隔一段 iteration 印出 loss
#%% Load image
import scipy.io
import scipy.linalg
#%%


fname2 = os.path.join(BASE_DIR, "HS Data", "Samson", "Y_clean.mat")
print("讀取資料路徑:", fname2)
mat2 = scipy.io.loadmat(fname2)
img_np_gt = mat2["Y_clean"]
img_np_gt = img_np_gt.transpose(2,0,1)
[p1, nr1, nc1] = img_np_gt.shape
#%%
# fname3  = "C:/Users/behnood/Desktop/VMCNN/Easy/A_true.mat"
# mat3 = scipy.io.loadmat(fname3)
# A_true_np = mat3["A_true"]
# A_true_np = A_true_np.transpose(2,0,1)
#%%
# fname4  = "C:/Users/behnood/Desktop/VMCNN/Easy/E.mat"
# mat4 = scipy.io.loadmat(fname4)
# E_np = mat4["E"]
rmax=3#E_np.shape[1] 

# 可選：如果資料夾內有 ground truth，程式會自動計算 RMSE / SAD。
# 沒有的話，至少會輸出 reconstruction RMSE。
def _load_optional_mat(path, key):
    if os.path.exists(path):
        mat = scipy.io.loadmat(path)
        if key in mat:
            return mat[key]
    return None

A_true_np = _load_optional_mat(os.path.join(BASE_DIR, "HS Data", "Samson", "A_true.mat"), "A_true")
if A_true_np is not None and A_true_np.ndim == 3:
    A_true_np = A_true_np.transpose(2, 0, 1)

E_true_np = _load_optional_mat(os.path.join(BASE_DIR, "HS Data", "Samson", "E.mat"), "E")

def abundance_rmse_percent(A_est, A_true):
    return 100.0 * np.sqrt(np.mean((A_est - A_true) ** 2))

def sad_matrix_degree(E_est, E_true):
    """
    計算 estimated endmembers 與 true endmembers 兩兩之間的 SAD。
    E_est : bands x r
    E_true: bands x r
    return: r x r SAD matrix，row=estimated，col=true
    """
    eps = 1e-12
    r_est = E_est.shape[1]
    r_true = E_true.shape[1]
    cost = np.zeros((r_est, r_true), dtype=np.float64)

    for i in range(r_est):
        for j in range(r_true):
            a = E_est[:, i]
            b = E_true[:, j]
            cosv = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)
            cosv = np.clip(cosv, -1.0, 1.0)
            cost[i, j] = np.degrees(np.arccos(cosv))
    return cost

def endmember_sad_degree(E_est, E_true):
    """直接依照目前順序計算 SAD。"""
    cost = sad_matrix_degree(E_est, E_true)
    r = min(E_est.shape[1], E_true.shape[1])
    sad_list = [cost[k, k] for k in range(r)]
    return float(np.mean(sad_list)), sad_list

def best_match_by_sad(E_est, A_est, E_true):
    """
    用 SAD 找最佳 endmember 配對，並同步重排 abundance channel。

    E_est : bands x r
    A_est : r x H x W
    E_true: bands x r

    return:
        E_matched: bands x r，已照 true endmember 順序排列
        A_matched: r x H x W，已照 true abundance 順序排列
        cost     : SAD matrix，row=estimated，col=true
        best_perm: true_index -> estimated_index 的對應
    """
    import itertools

    cost = sad_matrix_degree(E_est, E_true)
    r_true = E_true.shape[1]
    r_est = E_est.shape[1]

    best_perm = None
    best_score = np.inf

    # perm[true_idx] = est_idx
    for perm in itertools.permutations(range(r_est), r_true):
        score = sum(cost[perm[true_idx], true_idx] for true_idx in range(r_true))
        if score < best_score:
            best_score = score
            best_perm = perm

    E_matched = np.zeros_like(E_true)
    A_matched = np.zeros((r_true, A_est.shape[1], A_est.shape[2]), dtype=A_est.dtype)

    for true_idx, est_idx in enumerate(best_perm):
        E_matched[:, true_idx] = E_est[:, est_idx]
        A_matched[true_idx, :, :] = A_est[est_idx, :, :]

    return E_matched, A_matched, cost, best_perm

#%%

tol2=1
save_result=False
for fi in range(1):
    for fj in range(tol2):
            #%%
        #img_noisy_np = get_noisy_image(img_np_gt, 1/10)
        img_noisy_np = img_np_gt# add_noise(img_np_gt, 1/npar[0,fi])#11.55 20 dB, 36.7 30 dB, 116.5 40 dB
        #print(compare_snr(img_np_gt, img_noisy_np))
        img_resh=np.reshape(img_noisy_np,(p1,nr1*nc1))
        V, SS, U = scipy.linalg.svd(img_resh, full_matrices=False)
        PC=np.diag(SS)@U
        # img_resh_DN=V[:,:rmax]@PC[:rmax,:]
        img_resh_DN=V[:,:rmax]@V[:,:rmax].transpose(1,0)@img_resh
        img_resh_np_clip=np.clip(img_resh_DN, 0, 1)
        II,III = Endmember_extract(img_resh_np_clip,rmax)
        E_np1=img_resh_np_clip[:,II]
        #%% Set up Simulated 
        INPUT = 'noise' # 'meshgrid'
        pad = 'reflection'
        need_bias=True
        OPT_OVER = 'net' 
        
        # 
        LR1 = 0.001
        show_every = 100
        exp_weight=0.99
        
        num_iter1 = 8000
        input_depth =  img_noisy_np.shape[0]
        class CAE_EndEst(nn.Module):
            def __init__(self):
                super(CAE_EndEst, self).__init__()
                self.conv1 = nn.Sequential(
                    conv(input_depth, 256,3,1,bias=need_bias, pad=pad),
                    nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(0.1, inplace=True),
                )
                self.conv2 = nn.Sequential(
                    conv(256, 256,3,1,bias=need_bias, pad=pad),
                    nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(0.1, inplace=True),
                )
                self.conv3 = nn.Sequential(
                    conv(input_depth, 4, 1,1,bias=need_bias, pad=pad),
                    nn.BatchNorm2d(4,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(0.1, inplace=True),
                )
                self.dconv2 = nn.Sequential(
                    nn.Upsample(scale_factor=1),
                    conv(260, 256, 3,1,bias=need_bias, pad=pad),
                    nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(0.1, inplace=True),
                )
        
                self.dconv3 = nn.Sequential(
                    nn.Upsample(scale_factor=1),
                    conv(256, rmax, 3,1,bias=need_bias, pad=pad),
                    nn.BatchNorm2d(rmax,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Softmax(),
                )
                self.dconv4 = nn.Sequential(
                    nn.Linear(rmax, p1,bias=False),
                )
            def forward(self, x):
                x1 = self.conv3(x)
                x = self.conv1(x)
                x = torch.cat([x,x1], 1)
                x = self.dconv2(x)
                x2 = self.dconv3(x)
                x3 = torch.transpose(x2.view((rmax,nr1*nc1)),0,1)
                x3 = self.dconv4(x3)
                return x2,x3

        net1 = CAE_EndEst()
        net1.cuda()
        
        # Loss
        def my_loss(target, End2, lamb, out_):
            loss1 = 0.5*torch.norm((out_.transpose(1,0).view(1,p1,nr1,nc1) - target), 'fro')**2
            O = torch.mean(target.view(p1,nr1*nc1),1).type(dtype).view(p1,1)
            B = torch.from_numpy(np.identity(rmax)).type(dtype)
            loss2 = torch.norm(torch.mm(End2,B.view((rmax,rmax)))-O, 'fro')**2
            return loss1+lamb*loss2
        img_noisy_torch = torch.from_numpy(img_resh_DN).view(1,p1,nr1,nc1).type(dtype)
        net_input1 = get_noise(input_depth, INPUT,
            (img_noisy_np.shape[1], img_noisy_np.shape[2])).type(dtype).detach()
        E_torch = torch.from_numpy(E_np1).type(dtype)
        #%%
        # net_input_saved = net_input1.detach().clone()
        # noise = net_input1.detach().clone()
        out_avg = True
        
        i = 0
        loss_history = []
        def closure1():
            
            global i, out_LR_np, out_avg, out_avg_np, Eest
            
            out_LR,out_spec = net1(net_input1)
#            out_HR=torch.mm(E_torch.view(p1,rmax),out_LR.view(rmax,nr1*nc1))
            # Smoothing
            if out_avg is None:
                out_avg = out_LR.detach()
                # out_HR_avg = out_HR.detach()
            else:
                out_avg = out_avg * exp_weight + out_LR.detach() * (1 - exp_weight)
                # out_HR_avg = out_HR_avg * exp_weight + out_HR.detach() * (1 - exp_weight)

        #%%
            total_loss = my_loss(img_noisy_torch, net1.dconv4[0].weight,100,out_spec)
            loss_history.append(float(total_loss.item()))
            total_loss.backward()
         
          

            # print ('Iteration %05d    Loss %f   RMSE_LR: %f   RMSE_LR_avg: %f  SRE: %f SRE_avg: %f' % (i, total_loss.item(), RMSE_LR, RMSE_LR_avg, SRE, SRE_avg), '\r', end='')
            if  PLOT and i % show_every == 0:
                out_LR_np = out_LR.detach().cpu().squeeze().numpy()
                out_avg_np = out_avg.detach().cpu().squeeze().numpy()
                out_LR_np = np.clip(out_LR_np, 0, 1)
                out_avg_np = np.clip(out_avg_np, 0, 1)    
                f, ((ax1, ax2)) = plt.subplots(1, 2, sharey=True, figsize=(10,10))
                ax1.imshow(np.stack((out_LR_np[2,:,:],out_LR_np[1,:,:],out_LR_np[0,:,:]),2))
                ax2.imshow(np.stack((out_avg_np[2,:,:],out_avg_np[1,:,:],out_avg_np[0,:,:]),2))
                plt.show()                 
            i += 1       
            return total_loss
        net1.dconv4[0].weight=torch.nn.Parameter(E_torch.view(p1,rmax))       
        p11 = get_params(OPT_OVER, net1, net_input1)
        optimizer = torch.optim.Adam(p11, lr=LR1, betas=(0.9, 0.999), eps=1e-8,
                  weight_decay= 0, amsgrad=False)
        for j in range(num_iter1):
                optimizer.zero_grad()
                loss_value = closure1()  
                optimizer.step()
                if PRINT_PROGRESS and (j + 1) % 500 == 0:
                    print(f"Iteration {j+1:05d}/{num_iter1}, Loss = {loss_value.item():.6f}")
                net1.dconv4[0].weight.data[net1.dconv4[0].weight <= 0] = 0
                net1.dconv4[0].weight.data[net1.dconv4[0].weight >= 1] = 1
                if j>0:
                  Eest=net1.dconv4[0].weight.detach().cpu().squeeze().numpy()
                  if PLOT and j % show_every== 0: 
                    plt.plot(Eest)
                    plt.show()
                  
        out_avg_np = out_avg.detach().cpu().squeeze().numpy()
        out_avg_np = np.clip(out_avg_np, 0, 1)
        Eest = net1.dconv4[0].weight.detach().cpu().squeeze().numpy()

        # =========================
        # Final evaluation metrics
        # =========================
        A_est = out_avg_np.reshape(rmax, nr1 * nc1)
        Y_recon = Eest @ A_est
        recon_rmse = np.sqrt(np.mean((Y_recon - img_resh_DN) ** 2))
        recon_rmse_percent = recon_rmse * 100.0

        print("\n========== Final Result Metrics ==========")
        print(f"Dataset              : Samson")
        print(f"Image shape          : bands={p1}, height={nr1}, width={nc1}")
        print(f"Endmembers r         : {rmax}")
        print(f"Final loss           : {loss_history[-1]:.6f}")
        print(f"Reconstruction RMSE  : {recon_rmse:.8f}")
        print(f"Reconstruction RMSE% : {recon_rmse_percent:.4f}%")

        abundance_sum_error = np.mean(np.abs(np.sum(out_avg_np, axis=0) - 1.0))
        print(f"Abundance sum error  : {abundance_sum_error:.8f}")

        if E_true_np is not None and E_true_np.shape == Eest.shape:
            # 1) Direct evaluation：未重新排序，僅供比較
            mean_sad_direct, sad_each_direct = endmember_sad_degree(Eest, E_true_np)
            print(f"Endmember SAD direct : {mean_sad_direct:.4f} degree")
            print("SAD direct each      : " + ", ".join([f"{v:.4f}" for v in sad_each_direct]))

            # 2) Matched evaluation：用 SAD 找最佳端元配對，再同步重排 abundance
            Eest_matched, Aest_matched, sad_cost, best_perm = best_match_by_sad(
                Eest, out_avg_np, E_true_np
            )
            mean_sad_matched, sad_each_matched = endmember_sad_degree(Eest_matched, E_true_np)

            print("\n========== Matching Information ==========")
            print("SAD matrix: estimated endmember rows x true endmember columns")
            print(np.round(sad_cost, 4))
            print("Best matching: true_index <- estimated_index")
            for true_idx, est_idx in enumerate(best_perm):
                print(f"True {true_idx} <- Estimated {est_idx}, SAD = {sad_cost[est_idx, true_idx]:.4f} degree")
            print("==========================================")

            print(f"Endmember SAD matched : {mean_sad_matched:.4f} degree")
            print("SAD matched each      : " + ", ".join([f"{v:.4f}" for v in sad_each_matched]))

            if A_true_np is not None and A_true_np.shape == Aest_matched.shape:
                armse_direct = abundance_rmse_percent(out_avg_np, A_true_np)
                armse_matched = abundance_rmse_percent(Aest_matched, A_true_np)
                print(f"Abundance RMSE direct : {armse_direct:.4f}%")
                print(f"Abundance RMSE matched: {armse_matched:.4f}%")
            else:
                print("Abundance RMSE        : N/A，找不到或尺寸不符合 HS Data/Samson/A_true.mat")
        else:
            print("Endmember SAD         : N/A，找不到或尺寸不符合 HS Data/Samson/E.mat")
            if A_true_np is not None and A_true_np.shape == out_avg_np.shape:
                armse_direct = abundance_rmse_percent(out_avg_np, A_true_np)
                print(f"Abundance RMSE direct : {armse_direct:.4f}%")
            else:
                print("Abundance RMSE        : N/A，找不到或尺寸不符合 HS Data/Samson/A_true.mat")
        print("==========================================\n")

    #%%
        if  save_result is True:
                  scipy.io.savemat("Result/EestdB%01d%01d.mat" % (fi+2, fj+1),
                                    {'Eest%01d%01d' % (fi+2, fj+1):Eest})
                  scipy.io.savemat("Result/out_avg_npdB%01d%01d.mat" % (fi+2, fj+1),
                                    {'out_avg_np%01d%01d' % (fi+2, fj+1):out_avg_np.transpose(1,2,0)})
        #
