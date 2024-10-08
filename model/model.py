import math
from collections import OrderedDict

import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from timm.models.layers import trunc_normal_
from torch.nn import init

from CRSL import Complementary_Learning_Loss
from resnet import resnet50


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

        # self.dwconv = DWConv(hidden_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        # x = self.dwconv(x.permute(0,3,1,2)).permute(0,2,3,1)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )
        self.norm2 = nn.LayerNorm(dim)
        # self.mamba = VSSBlock(dim)
        self.mlp = FFN(dim, dim)
        # self.mlp = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(dim, dim * 4)),
        #     ("gelu", QuickGLUE()),
        #     ("c_proj", nn.Linear(dim * 4, dim))
        # ]))
    def forward(self, x):
        # print('x',x.shape)
        B, L, C = x.shape
        # batch_size, C, H, W = x.shape
        # x_flat = x.reshape(batch_size, C, -1).permute(0, 2, 1)
        x_flat = x.permute(0,2,1)
        # x_norm = self.norm1(x_flat)
        # x_mamba = self.mamba(x_norm)
        # 残差连接
        residual_output = x_flat + self.mamba(self.norm1(x_flat))
        residual_norm = self.mlp(self.norm2(residual_output))

        # out = self.mlp(residual_norm)
        # out = out.permute(0, 2, 1)
        # out = out.reshape(batch_size, C, H, W)

        #
        # x = x.permute(0, 2, 3, 1)
        # x = self.mamba(x)
        # out = x.permute(0, 3, 1, 2)
        return residual_norm.permute(0, 2, 1)

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        # self.inter_channels = reduc_ratio//reduc_ratio
        self.inter_channels = in_channels // reduc_ratio
        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype

        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGLUE(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class layerATT(nn.Module):
    # 1024,512
    def __init__(self, channel, edim, redic, channel_layer, H, W):
        super(layerATT, self).__init__()
        self.channel = channel
        self.channel_layer = channel_layer
        self.ATT_narrow = nn.Conv2d(self.channel, self.channel // 4, kernel_size=1, stride=1, padding=0)
        self.ATT_broad = nn.Conv2d(self.channel // 4, self.channel, kernel_size=1, stride=1, padding=0)
        self.mamba_broad = nn.Conv2d(self.channel // 4, self.channel, kernel_size=1, stride=1, padding=0)
        self.attn = nn.MultiheadAttention(embed_dim=edim, num_heads=edim // redic, batch_first=True)
        self.ln_1 = LayerNorm(edim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(edim, edim * 4)),
            ("gelu", QuickGLUE()),
            ("c_proj", nn.Linear(edim * 4, edim))
        ]))
        self.ln_2 = LayerNorm(edim)
        self.ln_1o = LayerNorm(edim)

        self.ATT_narrow1 = nn.Conv2d(self.channel_layer, self.channel // 4, kernel_size=(6, 6), stride=(4, 4),
                                     padding=(2, 2))
        self.ATT_narrow2 = nn.Conv2d(self.channel_layer, self.channel // 4, kernel_size=(4, 4), stride=(2, 2),
                                     padding=(1, 1))
        self.H = H
        self.W = W

        self.mambablock = MambaLayer(dim=self.channel // 4, d_state=64, d_conv=4, expand=2)
        self.mambablock_2 = MambaLayer(dim=self.channel // 4, d_state=64, d_conv=4, expand=2)
        # self.mamba = Mamba(
        #     d_model=self.channel // 2,  # Model dimension d_model
        #     d_state=16,  # SSM state expansion factor
        #     d_conv=4,  # Local convolution width
        #     expand=2  # Block expansion factor
        # )
        #
        # self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, x_ori, layer):
        if layer == 3:
            # 24,12
            x_ori = self.ATT_narrow(x_ori)
            x = self.ATT_narrow(x)

            channel = x.size(1)
            #
            x = x.view(x.size(0), channel, -1)
            x_ori = x_ori.view(x.size(0), channel, -1)

            x = self.ln_1(x)
            x_ori = self.ln_1o(x_ori)
            # x = x + self.mambablock(x)
            x = x + 0.1 * x_ori
            x = x + self.attn(x, x_ori, x_ori, need_weights=False)[0]
            # x = torch.cat((x, x_ori), dim=1)
            # x = x + self.mambablock(x)
            x = x + self.mambablock_2(x)
            # x = x + self.mlp(self.ln_2(x))
            x = x.view(x.size(0), channel, self.H, self.W)
            # x = self.ATT_broad(x)
            x = self.mamba_broad(x)
            return x
        if layer == 0 or layer == 1:
            # 96,48

            x_ori = self.ATT_narrow1(x_ori)
            x = self.ATT_narrow(x)
            channel = x.size(1)
            x = x.view(x.size(0), channel, -1)
            x_ori = x_ori.view(x.size(0), channel, -1)
            x = self.ln_1(x)
            x_ori = self.ln_1o(x_ori)
            # Inter-modal Enhanced Fusion
            x = x + 0.1 * x_ori
            x = x + self.attn(x, x_ori, x_ori, need_weights=False)[0]
            # x = torch.cat((x, x_ori), dim=1)
            # x = x + self.mambablock(x)
            x = self.mambablock(x)
            #
            # # tmp = self.mamba(x.permute(0,2,1), x_ori.permute(0,2,1))
            # # x = x + 0.1 * x_ori
            # x = x + self.attn(x_ori, x_ori, x, need_weights=False)[0]
            # x = x + self.mlp(self.ln_2(x))
            # x = self.ssm(x)
            x = x.view(x.size(0), channel, self.H, self.W)
            x = self.mamba_broad(x)
            return x

        if layer == 2:
            ##lay2 torch.Size([batch, 512, 54, 27])
            x_ori = self.ATT_narrow2(x_ori)
            x = self.ATT_narrow(x)
            channel = x.size(1)
            x = x.view(x.size(0), channel, -1)
            x_ori = x_ori.view(x.size(0), channel, -1)
            x = self.ln_1(x)
            x_ori = self.ln_1o(x_ori)
            x = x + 0.1 * x_ori
            x = x + self.attn(x, x_ori, x_ori, need_weights=False)[0]

            # x = torch.cat((x, x_ori), dim=1)
            x = self.mambablock(x)
            # x = x + self.attn(x_ori, x_ori, x, need_weights=False)[0]
            # x = x + self.mlp(self.ln_2(x))
            # x = self.ssm(x)
            x = x.view(x.size(0), channel, self.H, self.W)
            x = self.mamba_broad(x)
            return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v
        # self.conv1 = nn.Conv2d(64, 64, kernel_size=7,padding=3,bias=False)

    def forward(self, x):
        x = self.visible.conv1(x)
        # x = self.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t
        # self.conv1 = nn.Conv2d(64, 64, kernel_size=7,padding=3,bias=False)

    def forward(self, x):
        x = self.thermal.conv1(x)
        # x = self.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class visible_moduleA(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_moduleA, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v
        # self.conv1 = nn.Conv2d(64, 64, kernel_size=7,padding=3,bias=False)

    def forward(self, x):
        x = self.visible.conv1(x)
        # x = self.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_moduleA(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_moduleA, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t
        # self.conv1 = nn.Conv2d(64, 64, kernel_size=7,padding=3,bias=False)

    def forward(self, x):
        x = self.thermal.conv1(x)
        # x = self.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class embed_net(nn.Module):
    def __init__(self, class_num, no_local='on', gm_pool='on', arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.thermal_moduleA = thermal_moduleA(arch=arch)
        self.visible_moduleA = visible_moduleA(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.non_local = no_local
        if self.non_local == 'on':
            layers = [3, 4, 6, 3]
            non_layers = [0, 2, 3, 0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1)
                                    for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1)
                                    for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1)
                                    for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1)
                                    for i in range(non_layers[3])])

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.gm_pool = gm_pool
        dropout = 0
        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)
        out_planes = 64

        self.local_conv = nn.Conv2d(
            out_planes, 128, kernel_size=1, padding=0, bias=False)
        init.kaiming_normal(self.local_conv.weight, mode='fan_out')
        # may not be used, not working on caffe
        self.feat_bn2d = nn.BatchNorm2d(128)
        # initialize BN, may not be used
        init.constant(self.feat_bn2d.weight, 1)
        # iniitialize BN, may not be used
        init.constant(self.feat_bn2d.bias, 0)
        self.max_pool_PCB = nn.AdaptiveMaxPool2d((1, 1))

        self.layerATT0 = layerATT(channel=1024, edim=288, redic=9, channel_layer=64, H=24, W=12)
        self.layerATT1 = layerATT(channel=1024, edim=288, redic=9, channel_layer=256, H=24, W=12)
        self.layerATT2 = layerATT(channel=1024, edim=288, redic=9, channel_layer=512, H=24, W=12)
        self.layerATT3 = layerATT(channel=1024, edim=288, redic=9, channel_layer=1024, H=24, W=12)

    def forward(self, x1_1, x1_2, x2_1, x2_2, modal=0):
        if modal == 0:
            x1_1 = self.visible_module(x1_1)
            x2_1 = self.thermal_module(x2_1)

            x1_2 = self.visible_moduleA(x1_2)
            x2_2 = self.thermal_moduleA(x2_2)

            x = torch.cat((x1_1, x1_2, x2_1, x2_2), 0)
            # x = torch.cat((x1_1, x2_1), 0)
        elif modal == 1:
            x1_1 = self.visible_module(x1_1)
            x1_2 = self.visible_moduleA(x1_2)
            x_mix = (x1_1 + x1_2) / 2
            # x_mix = x1_1
            # x_mix = x1_2
            x = x_mix

            # x = torch.cat((x1_1, x1_1), 0)
            # x = torch.cat((x1_2, x1_2), 0)
        elif modal == 2:
            x2_1 = self.thermal_module(x2_1)
            x2_2 = self.thermal_moduleA(x2_2)
            x_mix = (x2_1 + x2_2) / 2
            # x_mix = x2_1
            # x_mix = x2_2
            x = x_mix

            # x = torch.cat((x2_1, x2_1), 0)
            # x = torch.cat((x2_2, x2_2), 0)

        # shared block
        if self.non_local == 'on':
            # torch.Size([16, 64, 108, 54])
            xlay0 = x
            NL1_counter = 0
            if len(self.NL_1_idx) == 0:
                self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            # Layer 2
            # #([16, 256, 108, 54])
            xlay1 = x
            NL2_counter = 0
            if len(self.NL_2_idx) == 0:
                self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            # torch.Size([16, 512, 54, 27])
            xlay2 = x
            NL3_counter = 0
            if len(self.NL_3_idx) == 0:
                self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # Layer 4

            xlay3 = x
            x = self.layerATT0(x, xlay0, 0)
            x = self.layerATT1(x, xlay1, 1)
            x = self.layerATT2(x, xlay2, 2)
            x = self.layerATT3(x, xlay3, 3)
            NL4_counter = 0
            if len(self.NL_4_idx) == 0:
                self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:

            x = self.base_resnet.base.layer1(x)
            x = self.base_resnet.base.layer2(x)
            x = self.base_resnet.base.layer3(x)
            x = self.base_resnet.base.layer4(x)

        # x = self.eablock(x)
        x = x.float()
        xlay4 = x
        if self.gm_pool == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool)

        if self.training:
            # losss1 = self.complementary_Learning_Loss(xlay1)
            # losss2 = self.complementary_Learning_Loss(xlay2)
            # losss3 = self.complementary_Learning_Loss(xlay3)
            losss4 = self.complementary_Learning_Loss(xlay4)
            # loss = losss1 + losss2 + losss3 + losss4
            loss = losss4
            return x_pool, self.classifier(feat), loss

        else:

            return self.l2norm(x_pool), self.l2norm(feat)

    def complementary_Learning_Loss(self, xlay):
        x_sid_rgb, x_sid_ir = torch.chunk(xlay, 2, 0)
        loss = Complementary_Learning_Loss(x_sid_rgb, x_sid_ir).cuda()
        loss_mean = torch.mean(loss)
        loss_std = torch.std(loss)
        loss_new = (loss - loss_mean) / loss_std
        loss = torch.mean(loss_new)
        return loss

