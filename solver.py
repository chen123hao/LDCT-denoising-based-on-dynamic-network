import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from prep import printProgressBar
from networks import RED_CNN
from networks import CTformer
from networks import ch_net
from measure import compute_measure


class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader
        self.load_pre = args.load_pre

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size
#############################################################################################################################
        self.REDCNN = RED_CNN()
        self.REDCNN.to(self.device)
        self.REDCNN.load_state_dict(torch.load("/home/ch/bmecode/ldct_denoising/RED-CNN-master/5save/REDCNN_4000iter.ckpt"))
#############################################################################################################################
        self.CTFormer = CTformer(img_size=64, tokens_type='performer', embed_dim=64, depth=1, num_heads=8, kernel=8,
                                 stride=4, mlp_ratio=2., token_dim=64)
        self.CTFormer.to(self.device)
        self.CTFormer.load_state_dict(torch.load("/home/ch/bmecode/ldct_denoising/CTformer-main/5save/T2T_vit_31936iter.ckpt",
                                                                 map_location='cpu'))
#############################################################################################################################







        self.net = ch_net(block1=self.REDCNN, block2=self.CTFormer).to(self.device)
        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.REDCNN.parameters(), self.lr)


    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'ch_net2_gate3_{}iter.ckpt'.format(iter_))
        torch.save(self.net.state_dict(), f)###########################################################


    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.REDCNN.load_state_dict(state_d)
        else:
            self.REDCNN.load_state_dict(torch.load(f))


    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()


    def train(self):
        if self.load_pre:
            self.net = ch_net(block1=self.REDCNN, block2=self.CTFormer).to(self.device)
            self.net.load_state_dict(torch.load("/home/ch/bmecode/plan2/3save/ch_net2_res18gate1_3000iter.ckpt"))
        for name, p in self.net.named_parameters():
            if name.startswith('block'):
                p.requires_grad = False
        train_losses = []
        total_iters = 0
        start_time = time.time()
        for epoch in range(1, self.num_epochs):
            self.net.train(True)

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)


                if  self.patch_size: # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)#裁剪后量变为原来的64倍（8*8)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                pred,mask,prob = self.net(x)
                ########################################################################
                # if self.load_pre == True and mask == 1:
                #     pred = 50*pred
                ########################################################################
                loss = self.criterion(pred, y)
                self.net.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.data_loader), loss.item(), 
                                                                                                        time.time() - start_time))
                    print(mask)
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))
        self.save_model(total_iters)#############################################################################################

    def test(self):
        del self.net
        # load

        self.net = ch_net(block1=self.REDCNN, block2=self.CTFormer).to(self.device)
        self.net.load_state_dict(torch.load("/home/ch/bmecode/plan2/4save/ch_net2_gate3_999iter.ckpt"))######### 名字未改 ####################
        self.net.eval()
        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):

                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                pred,mask,prob = self.net(x)
                if mask==0:
                    print("yahou")
                print(mask)




                # denormalize, truncate
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                # save result figure##################################################################################################
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)

                printProgressBar(i, len(self.data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
            print('\n')
            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader),
                                                                                            ori_ssim_avg/len(self.data_loader),
                                                                                            ori_rmse_avg/len(self.data_loader)))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader),
                                                                                                  pred_ssim_avg/len(self.data_loader),
                                                                                                  pred_rmse_avg/len(self.data_loader)))
            print('\n')
