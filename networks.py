import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable
import torch.autograd as autograd
import torchvision.transforms as tfs
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from token_transformer import Token_transformer
from token_performer import Token_performer
from T2T_transformer_block import Block, get_sinusoid_encoding

#################
###  BLOCK1  ####
#################
class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out

#################
###  BLOCK2  ####
#################
class MultiHeadDense(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ch, out_ch))

    def forward(self, x):
        # x:[b, h*w, d]
        # x = torch.bmm(x, self.weight)
        x = F.linear(x, self.weight)
        return x


class T2T_module(nn.Module):
    """
    CTformer encoding module
    """

    def __init__(self, img_size=64, tokens_type='performer', in_chans=1, embed_dim=256, token_dim=64, kernel=32,
                 stride=32):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            # print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1))

            self.attention1 = Token_performer(dim=in_chans * 7 * 7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim * 3 * 3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)
        # self.num_patches = (img_size // (1 * 2 * 2)) * (img_size // (1 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately
        self.num_patches = 529  ## calculate myself

    def forward(self, x):

        # Tokenization
        x = self.soft_split0(x)  ## [1, 128, 64, 128])

        # CTformer module A
        x = self.attention1(x.transpose(1, 2))
        res_11 = x
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x = torch.roll(x, shifts=(2, 2), dims=(2, 3))  ##  shift some position
        x = self.soft_split1(x)

        # CTformer module B
        x = self.attention2(x.transpose(1, 2))
        res_22 = x
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x = torch.roll(x, shifts=(2, 2), dims=(2, 3))  ## shift back position
        x = self.soft_split2(x)

        x = self.project(x.transpose(1, 2))  ## no projection
        return x, res_11, res_22  # ,res0,res2


class Token_back_Image(nn.Module):
    """
    CTformer decoding module
    """

    def __init__(self, img_size=64, tokens_type='performer', in_chans=1, embed_dim=256, token_dim=64, kernel=32,
                 stride=32):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Fold((64, 64), kernel_size=(7, 7), stride=(2, 2))
            self.soft_split1 = nn.Fold((29, 29), kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2))
            self.soft_split2 = nn.Fold((25, 25), kernel_size=(3, 3), stride=(1, 1))

            self.attention1 = Token_transformer(dim=token_dim, in_dim=in_chans * 7 * 7, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim, in_dim=token_dim * 3 * 3, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(embed_dim, token_dim * 3 * 3)
        elif tokens_type == 'performer':
            # print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Fold((64, 64), kernel_size=(7, 7), stride=(2, 2))
            self.soft_split1 = nn.Fold((29, 29), kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2))
            self.soft_split2 = nn.Fold((25, 25), kernel_size=(3, 3), stride=(1, 1))

            self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans * 7 * 7, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim * 3 * 3, kernel_ratio=0.5)
            self.project = nn.Linear(embed_dim, token_dim * 3 * 3)

        self.num_patches = (img_size // (1 * 2 * 2)) * (
                    img_size // (1 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x, res_11, res_22):
        x = self.project(x).transpose(1, 2)

        # CTformer module C
        x = self.soft_split2(x)
        x = torch.roll(x, shifts=(-2, -2), dims=(-1, -2))
        x = rearrange(x, 'b c h w -> b c (h w)').transpose(1, 2)
        x = x + res_22
        x = self.attention2(x).transpose(1, 2)

        # CTformer module D
        x = self.soft_split1(x)
        x = torch.roll(x, shifts=(-2, -2), dims=(-1, -2))
        x = rearrange(x, 'b c h w -> b c (h w)').transpose(1, 2)
        x = x + res_11
        x = self.attention1(x).transpose(1, 2)

        # Detokenization
        x = self.soft_split0(x)

        return x


class CTformer(nn.Module):
    def __init__(self, img_size=512, tokens_type='convolution', in_chans=1, num_classes=1000, embed_dim=768, depth=12,
                 ## transformer depth 12
                 num_heads=12, kernel=32, stride=32, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.1,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, token_dim=1024):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(  ## use module 2
            img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim,
            kernel=kernel, stride=stride)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim),
                                      requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # CTformer decoder
        self.dconv1 = Token_back_Image(img_size=img_size, tokens_type=tokens_type, in_chans=in_chans,
                                       embed_dim=embed_dim, token_dim=token_dim, kernel=kernel, stride=stride)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        res1 = x
        x, res_11, res_22 = self.tokens_to_token(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        i = 0
        for blk in self.blocks:  ## only one intermediate transformer block
            i += 1
            x = blk(x)

        x = self.norm(x)  # + res_0   ## do not use 0,2,4
        out = res1 - self.dconv1(x, res_11, res_22)
        return out
def split_arr(arr,patch_size,stride=32):    ## 512*512 to 32*32
    pad = (16, 16, 16, 16) # pad by (0, 1), (2, 1), and (3, 3)
    arr = nn.functional.pad(arr, pad, "constant", 0)
    _,_,h,w = arr.shape
    num = h//stride - 1
    arrs = torch.zeros(num*num,1,patch_size,patch_size)

    for i in range(num):
        for j in range(num):
            arrs[i*num+j,0] = arr[0,0,i*stride:i*stride+patch_size,j*stride:j*stride+patch_size]
    return arrs

def agg_arr(arrs, size, stride=32):  ## from 32*32 to size 512*512
    arr = torch.zeros(size, size)
    n,_,h,w = arrs.shape
    num = size//stride
    for i in range(num):
        for j in range(num):
            arr[i*stride:(i+1)*stride,j*stride:(j+1)*stride] = arrs[i*num+j,:,16:48,16:48]
  #return arr
    return arr.unsqueeze(0).unsqueeze(1)

#################
###   GATE1  ####
#################
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class FeedforwardGateI(nn.Module):
    """ Use Max Pooling First and then apply to multiple 2 conv layers.
    The first conv has stride = 1 and second has stride = 2,(加一层1*1卷积来改变通道数)"""
    def __init__(self, channel=10):
        super(FeedforwardGateI, self).__init__()
        #self.pool_size = pool_size
        self.channel = channel
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=1, stride=1, padding=0 ,bias=False)
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = conv3x3(channel, channel)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        # adding another conv layer
        self.conv2 = conv3x3(channel, channel, stride=2)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU(inplace=True)

        #pool_size = math.floor(pool_size/2)  # for max pooling
        #pool_size = math.floor(pool_size/2 + 0.5)  # for conv stride = 2

        #self.avg_layer = nn.AvgPool2d(pool_size)#pool_size = 1
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))################输出1*1
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv0(x)
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.global_pool(x)
        batch = x.size(0)
        x = self.linear_layer(x).squeeze()

        softmax = self.prob_layer(x)###################################
        logprob = self.logprob(x)

        #单张测试时防size(0)消失
        if softmax.size(0)!=batch:
            softmax = softmax.unsqueeze(dim=0)

        x = (softmax[:, 1] > 0.5).float().detach() - \
            softmax[:, 1].detach() + softmax[:, 1]
        x = x.view(batch, 1, 1, 1)
        return x, logprob

#################
###   GATE2  ####
#################
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):



        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

gate2 = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 2))
class Gate2(nn.Module):
    def __init__(self,gate=gate2):
        super(Gate2, self).__init__()
        self.net = gate
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
    def forward(self,x):
        batch = x.size(0)
        x = self.net(x).squeeze()
        softmax = self.prob_layer(x)
        logprob = self.logprob(x)

        #单张测试时防size(0)消失
        if softmax.size(0)!=batch:
            softmax = softmax.unsqueeze(dim=0)

        x = (softmax[:, 1] > 0.5).float().detach() - \
            softmax[:, 1].detach() + softmax[:, 1]
        x = x.view(batch, 1, 1, 1)
        return x, logprob

#################
###   GATE3  ####
#################
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Gate3(nn.Module):
    """ Use Max Pooling First and then apply to multiple 2 conv layers.
    The first conv has stride = 1 and second has stride = 2,(加一层1*1卷积来改变通道数)"""
    def __init__(self,input_chan):
        super(Gate3, self).__init__()
        #self.pool_size = pool_size

        self.conv0 = conv3x3(input_chan,64,stride=2)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = conv3x3(64,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)


        self.conv2 = conv3x3(64, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv3x3(128, 256, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = conv3x3(256, 512, stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)

        #pool_size = math.floor(pool_size/2)  # for max pooling
        #pool_size = math.floor(pool_size/2 + 0.5)  # for conv stride = 2

        #self.avg_layer = nn.AvgPool2d(pool_size)#pool_size = 1
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))################输出1*1
        self.linear_layer = nn.Conv2d(in_channels=512, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()



    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.global_pool(x)
        batch = x.size(0)
        x = self.linear_layer(x).squeeze()

        softmax = self.prob_layer(x)###################################
        logprob = self.logprob(x)


        #单张测试时防size(0)消失
        if softmax.size(0)!=batch:
            softmax = softmax.unsqueeze(dim=0)

        x = (softmax[:, 1] > 0.5).float().detach() - \
            softmax[:, 1].detach() + softmax[:, 1]
        x = x.view(batch, 1, 1, 1)
        return x, softmax








#################
###   MODEL  ####
#################
class ch_net(nn.Module):
    """ SkipNets with Feed-forward Gates for Supervised Pre-training stage.
    Adding one routing module after each basic block."""

    # def __init__(self,block1,block2, gate1=Gate2(gate=gate2),gate2=Gate2(gate=gate2)):
    #     super(ch_net, self).__init__()
    #
    #     self.block1 = block1
    #     self.block2 = block2
    #     self.gate1 = gate1
    #     self.gate2 = gate2
    #     #self.gate2 = gate2(channel=planes)
    #
    # def forward(self, x):
    #     """Return output logits, masks(gate ouputs) and probabilities
    #     associated to each gate."""
    #     masks = []
    #     gprobs = []
    #
    #     #gate1 takes the input
    #     mask, gprob = self.gate1(x)
    #     gprobs.append(gprob)
    #     masks.append(mask.squeeze())
    #     prev = x
    #     x = self.block1(x)
    #     x = mask.expand_as(x) * x + (1 - mask).expand_as(prev) * prev
    #
    #     #gate2
    #     mask, gprob = self.gate2(x)
    #     gprobs.append(gprob)
    #     masks.append(mask.squeeze())
    #     prev = x
    #
    #     #mask = 1-mask.data
    #     #masks.append(mask.squeeze())
    #     #prev = x
    #     if self.training:
    #         x = self.block2(x)
    #     else:
    #         arrs = split_arr(x, 64).to('cuda:1')
    #         arrs[0:64] = self.block2(arrs[0:64])
    #         arrs[64:2 * 64] = self.block2(arrs[64:2 * 64])
    #         arrs[2 * 64:3 * 64] = self.block2(arrs[2 * 64:3 * 64])
    #         arrs[3 * 64:4 * 64] = self.block2(arrs[3 * 64:4 * 64])
    #         x = agg_arr(arrs, 512).to('cuda:1')
    #     x = mask.expand_as(x) * x + (1 - mask).expand_as(prev) * prev
        #不必删去最后一个门的决策，因为ch用了每一个门
        ############################################################################################################################################################
    def __init__(self,block1,block2, gate=Gate3(1)):
        super(ch_net, self).__init__()

        self.block1 = block1
        self.block2 = block2
        self.gate = gate



    def forward(self, x):
        """Return output logits, masks(gate ouputs) and probabilities
        associated to each gate."""
        masks = []
        gprobs = []

        #gate takes the input
        mask, gprob = self.gate(x)
        gprobs.append(gprob)
        masks.append(mask.squeeze())

        x1 = self.block1(x)


        if self.training:
            x2 = self.block2(x)
        else:
            arrs = split_arr(x, 64).to('cuda:1')
            arrs[0:64] = self.block2(arrs[0:64])
            arrs[64:2 * 64] = self.block2(arrs[64:2 * 64])
            arrs[2 * 64:3 * 64] = self.block2(arrs[2 * 64:3 * 64])
            arrs[3 * 64:4 * 64] = self.block2(arrs[3 * 64:4 * 64])
            x2 = agg_arr(arrs, 512).to('cuda:1')
        x = mask.expand_as(x1) * x1 + (1 - mask).expand_as(x2) * x2############################################################
        ############################################################################################################################################################
        return x, masks, gprobs