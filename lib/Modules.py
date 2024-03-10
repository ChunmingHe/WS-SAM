# -*- coding: utf-8 -*-
import torch.nn as nn

affine_par = True
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import time
import einops
from torch.nn.parameter import Parameter
from thop import profile
from lib.Slot import SoftPositionEmbed,spatial_broadcast,spatial_flatten,spatial_broadcast2,unstack_and_split
# import numpy as np #借助numpy模块的set_printoptions()函数，将打印上限设置为无限即可
# np.set_printoptions(threshold=np.inf)
"""
    basic conv block
"""


# 低频分量使用 风格迁移的作用
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=1)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


class BasicDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, out_padding=0, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicDeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, output_padding=out_padding, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


"""
    position attention module
"""


class PAM(nn.Module):
    def __init__(self, in_channels):
        super(PAM, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out


"""
    global prior module
"""


# 目前太肥了，这玩意儿输出一个12*12的图片，需要2048的channel？
class GPM(nn.Module):
    def __init__(self, dilation_series=[6, 12, 18], padding_series=[6, 12, 18], depth=128):
        # def __init__(self, dilation_series=[2, 5, 7], padding_series=[2, 5, 7], depth=128):
        super(GPM, self).__init__()
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv2d(2048, depth, kernel_size=1, stride=1)
        )
        self.branch0 = BasicConv2d(2048, depth, kernel_size=1, stride=1)
        self.branch1 = BasicConv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[0],
                                   dilation=dilation_series[0])
        self.branch2 = BasicConv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[1],
                                   dilation=dilation_series[1])
        self.branch3 = BasicConv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[2],
                                   dilation=dilation_series[2])
        self.head = nn.Sequential(
            BasicConv2d(depth * 5, 256, kernel_size=3, padding=1),
            PAM(256)
        )
        self.out = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64, affine=affine_par),
            nn.PReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # out = self.conv2d_list[0](x)
        # mulBranches = [conv2d_l(x) for conv2d_l in self.conv2d_list]
        size = x.shape[2:]
        branch_main = self.branch_main(x)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        out = torch.cat([branch_main, branch0, branch1, branch2, branch3], 1)
        out = self.head(out)
        out = self.out(out)
        return out


"""
    enhance texture module
"""


class ETM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ETM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = BasicConv2d(in_channels, out_channels, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



"""
InterFA
"""


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class InterFA(nn.Module):
    def __init__(self, in_channels):
        super(InterFA, self).__init__()
        self.conv3x3 = BasicConv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.cbam = CBAM(in_channels)
        self.conv1x1 = BasicConv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f1, f2):
        f2_up = F.interpolate(f2, size=f1.size()[2:], mode='bilinear', align_corners=True)
        cat = torch.cat([f1, f2_up], dim=1)
        f = self.conv3x3(cat)
        f = self.cbam(f)
        cat2 = torch.cat([f, f1], dim=1)
        out = self.conv1x1(cat2)
        return f, out


class GCM_interFA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM_interFA, self).__init__()
        self.T1 = ETM(in_channels, out_channels)
        self.T2 = ETM(in_channels * 2, out_channels)
        self.T3 = ETM(in_channels * 4, out_channels)
        self.T4 = ETM(in_channels * 8, out_channels)
        self.interFA1 = InterFA(out_channels)
        self.interFA2 = InterFA(out_channels)
        self.interFA3 = InterFA(out_channels)

    def forward(self, f1, f2, f3, f4):
        f1 = self.T1(f1)  # 96,96,96
        f2 = self.T2(f2)  # 96,48,48
        f3 = self.T3(f3)  # 96,24,24
        f4 = self.T4(f4)  # 96,12,12
        temp3, f3 = self.interFA3(f3, f4)
        temp2, f2 = self.interFA2(f2, temp3)
        temp1, f1 = self.interFA1(f1, temp2)

        return f1, f2, f3, f4


"""
InterFA+ODE+slot
"""
class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_slots, encoder_dims, iters=3, hidden_dim=128, eps=1e-8):
        """Builds the Slot Attention module.
        Args:
            iters: Number of iterations.
            num_slots: Number of slots.
            encoder_dims: Dimensionality of slot feature vectors.
            hidden_dim: Hidden layer size of MLP.
            eps: Offset for attention coefficients before normalization.
        """
        super(SlotAttention, self).__init__()

        self.eps = eps
        self.iters = iters
        self.num_slots = num_slots
        self.scale = encoder_dims ** -0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)

        # Parameters for Gaussian init (shared by all slots).
        # self.slots_mu = nn.Parameter(torch.randn(1, 1, encoder_dims))
        # self.slots_sigma = nn.Parameter(torch.randn(1, 1, encoder_dims))

        self.slots_embedding = nn.Embedding(num_slots, encoder_dims)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(encoder_dims, encoder_dims)
        self.project_k = nn.Linear(encoder_dims, encoder_dims)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)

        # Slot update functions.
        self.gru = nn.GRUCell(encoder_dims, encoder_dims)

        hidden_dim = max(encoder_dims, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, encoder_dims)
        )

    def forward(self, inputs, num_slots=None):
        # inputs has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_input(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        # random slots initialization,
        # mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = self.slots_sigma.expand(b, n_s, -1)
        # slots = torch.normal(mu, sigma)

        # learnable slots initialization
        slots = self.slots_embedding(torch.arange(0, n_s).expand(b, n_s).to(self.device))

        # Multiple rounds of attention.
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.

            updates = torch.einsum('bjd,bij->bid', v, attn)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class SlotAttentionModule(nn.Module):
    def __init__(self, encoder_dims, resolution, num_slots, iters):
        super(SlotAttentionModule, self).__init__()
        self.resolution = resolution
        self.encoder_pos = SoftPositionEmbed(encoder_dims, ((int(resolution), int(resolution))))
        self.layer_norm = nn.LayerNorm(encoder_dims)
        self.mlp = nn.Sequential(
                nn.Linear(encoder_dims, encoder_dims),
                nn.ReLU(inplace=True),
                nn.Linear(encoder_dims, encoder_dims)
            )
        self.slot_attention = SlotAttention(iters=iters,
                                            num_slots=num_slots,
                                            encoder_dims=encoder_dims,
                                            hidden_dim=encoder_dims)
        self.decoder_pos = SoftPositionEmbed(encoder_dims, (int(resolution), int(resolution)))
        self.conv = nn.Conv2d(encoder_dims*num_slots, encoder_dims, kernel_size=1, padding=0, stride=1)
        
    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.encoder_pos(x) 
        x = spatial_flatten(x)
        x = self.mlp(self.layer_norm(x))
        slots = self.slot_attention(x)
        x = spatial_broadcast(slots, (int(self.resolution), int(self.resolution)))
        x = self.decoder_pos(x)
        x = einops.rearrange(x, 'b n h w c -> b (n c) h w')
        out = self.conv(x)
        return out


class SlotAttentionModule2(nn.Module):
    def __init__(self, encoder_dims, resolution, num_slots, iters):
        super(SlotAttentionModule2, self).__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.encoder_pos = SoftPositionEmbed(encoder_dims, ((int(resolution), int(resolution))))
        self.layer_norm = nn.LayerNorm(encoder_dims)
        self.mlp = nn.Sequential(
                nn.Linear(encoder_dims, encoder_dims),
                nn.ReLU(inplace=True),
                nn.Linear(encoder_dims, encoder_dims)
            )
        self.slot_attention = SlotAttention(iters=iters,
                                            num_slots=num_slots,
                                            encoder_dims=encoder_dims,
                                            hidden_dim=encoder_dims)
        self.decoder_pos = SoftPositionEmbed(encoder_dims, (int(resolution), int(resolution)))

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(encoder_dims, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(64, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(64, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3 + 1, kernel_size=5, padding=2, stride=1)
        )
        self.convModules = nn.ModuleList([nn.Conv2d(encoder_dims, encoder_dims//num_slots, kernel_size=1, padding=0, stride=1) for i in range (num_slots)])
        
    def forward(self, x):
        batch = x.shape[0]
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.encoder_pos(x) 
        x = spatial_flatten(x)
        x = self.mlp(self.layer_norm(x))
        slots = self.slot_attention(x)
        x = spatial_broadcast2(slots, (int(self.resolution), int(self.resolution)))
        y = spatial_broadcast(slots, (int(self.resolution), int(self.resolution)))
        x = self.decoder_pos(x)
        y = self.decoder_pos(y)
        z_conv = []
        for i in range(self.num_slots):
            z = y[:,i,:,:,:]
            z = einops.rearrange(z, 'b h w c -> b c h w')
            z1 = self.convModules[i](z)
            z_conv.append(z1)
        z = torch.stack(z_conv,1)
        z = einops.rearrange(z, 'b n c h w -> b (n c) h w')
        x = einops.rearrange(x, 'b_n h w c -> b_n c h w')
        x = self.decoder_cnn(x)
        recons, masks = unstack_and_split(x, batch_size=batch, num_channels=3)
        masks = torch.softmax(masks, axis=1)
        recon_combined = torch.sum(recons * masks, axis=1)  # Recombine image.
        return recon_combined, z


class getAlpha(nn.Module):
    def __init__(self, in_channels):
        super(getAlpha, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels*2,in_channels,kernel_size =1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels,1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ODE(nn.Module):
    def __init__(self, in_channels,num_slots_N,num_slots_M,iters,resolutions):
        super(ODE, self).__init__()
        self.SAn = SlotAttentionModule(in_channels, resolutions, num_slots_N, iters)
        self.SAm = SlotAttentionModule(in_channels, resolutions, num_slots_M, iters)
        self.getalpha = getAlpha(in_channels)

    def forward(self, feature_map):
        f1 = self.SAn(feature_map)
        f2 = self.SAm(f1+feature_map)
        alpha = self.getalpha(torch.cat([f1,f2],dim=1))
        out = feature_map+f1*alpha+f2*(1-alpha)
        return  out

class ODE2(nn.Module):
    def __init__(self, in_channels,num_slots_N,num_slots_M,iters,resolutions):
        super(ODE2, self).__init__()
        self.SAn = SlotAttentionModule2(in_channels, resolutions, num_slots_N, iters)
        self.SAm = SlotAttentionModule2(in_channels, resolutions, num_slots_M, iters)
        self.getalpha = getAlpha(in_channels)

    def forward(self, feature_map):
        recon_combined1,f1 = self.SAn(feature_map)
        recon_combined2,f2 = self.SAm(f1+feature_map)
        alpha = self.getalpha(torch.cat([f1,f2],dim=1))
        out = feature_map+f1*alpha+f2*(1-alpha)
        return  out,recon_combined1,recon_combined2


class GCM_interFA_ODE_slot(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM_interFA_ODE_slot, self).__init__()
        self.T1 = ETM(in_channels, out_channels)
        self.T2 = ETM(in_channels * 2, out_channels)
        self.T3 = ETM(in_channels * 4, out_channels)
        self.T4 = ETM(in_channels * 8, out_channels)
        self.ODE3 = ODE(out_channels,num_slots_N=4,num_slots_M=2,iters=3,resolutions=24)
        self.ODE4 = ODE(out_channels,num_slots_N=4,num_slots_M=2,iters=3,resolutions=12)
        self.interFA1 = InterFA(out_channels)
        self.interFA2 = InterFA(out_channels)
        self.interFA3 = InterFA(out_channels)

    def forward(self, f1, f2, f3, f4):
        f1 = self.T1(f1)  # 96,96,96
        f2 = self.T2(f2)  # 96,48,48
        f3 = self.T3(f3)  # 96,24,24
        f3 = self.ODE3(f3)
        f4 = self.T4(f4)  # 96,12,12
        f4 = self.ODE4(f4)
        temp3, f3 = self.interFA3(f3, f4)
        temp2, f2 = self.interFA2(f2, temp3)
        temp1, f1 = self.interFA1(f1, temp2)

        return f1, f2, f3, f4


class GCM_interFA_ODE_slot2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM_interFA_ODE_slot2, self).__init__()
        self.T1 = ETM(in_channels, out_channels)
        self.T2 = ETM(in_channels * 2, out_channels)
        self.T3 = ETM(in_channels * 4, out_channels)
        self.T4 = ETM(in_channels * 8, out_channels)
        self.ODE3 = ODE2(out_channels,num_slots_N=4,num_slots_M=2,iters=3,resolutions=24)
        self.ODE4 = ODE2(out_channels,num_slots_N=4,num_slots_M=2,iters=3,resolutions=12)
        self.interFA1 = InterFA(out_channels)
        self.interFA2 = InterFA(out_channels)
        self.interFA3 = InterFA(out_channels)

    def forward(self, f1, f2, f3, f4):
        f1 = self.T1(f1)  # 96,96,96
        f2 = self.T2(f2)  # 96,48,48
        f3 = self.T3(f3)  # 96,24,24
        f3,recon_combined31,recon_combined32 = self.ODE3(f3)
        f4 = self.T4(f4)  # 96,12,12
        f4,recon_combined41,recon_combined42 = self.ODE4(f4)
        temp3, f3 = self.interFA3(f3, f4)
        temp2, f2 = self.interFA2(f2, temp3)
        temp1, f1 = self.interFA1(f1, temp2)

        return f1, f2, f3, f4,recon_combined31,recon_combined32,recon_combined41,recon_combined42


"""
slot在哪个层的消融实验
"""
class GCM_interFA_ODE_slot_channel4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM_interFA_ODE_slot_channel4, self).__init__()
        self.T1 = ETM(in_channels, out_channels)
        self.T2 = ETM(in_channels * 2, out_channels)
        self.T3 = ETM(in_channels * 4, out_channels)
        self.T4 = ETM(in_channels * 8, out_channels)
        self.ODE4 = ODE(out_channels,num_slots_N=4,num_slots_M=2,iters=3,resolutions=12)
        self.interFA1 = InterFA(out_channels)
        self.interFA2 = InterFA(out_channels)
        self.interFA3 = InterFA(out_channels)

    def forward(self, f1, f2, f3, f4):
        f1 = self.T1(f1)  # 96,96,96
        f2 = self.T2(f2)  # 96,48,48
        f3 = self.T3(f3)  # 96,24,24
        f4 = self.T4(f4)  # 96,12,12
        f4 = self.ODE4(f4)
        temp3, f3 = self.interFA3(f3, f4)
        temp2, f2 = self.interFA2(f2, temp3)
        temp1, f1 = self.interFA1(f1, temp2)

        return f1, f2, f3, f4

class GCM_interFA_ODE_slot_channel4_scribble(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM_interFA_ODE_slot_channel4_scribble, self).__init__()
        self.T1 = ETM(in_channels, out_channels)
        self.T2 = ETM(in_channels * 2, out_channels)
        self.T3 = ETM(in_channels * 4, out_channels)
        self.T4 = ETM(in_channels * 8, out_channels)
        self.ODE4 = ODE(out_channels,num_slots_N=4,num_slots_M=2,iters=3,resolutions=12)
        self.interFA1 = InterFA(out_channels)
        self.interFA2 = InterFA(out_channels)
        self.interFA3 = InterFA(out_channels)
        mid_channel = 256
        self.embedding_layer = nn.Sequential(
            nn.Conv2d(out_channels*4, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

    def forward(self, f1, f2, f3, f4):
        f1 = self.T1(f1)  # 96,96,96
        f2 = self.T2(f2)  # 96,48,48
        f3 = self.T3(f3)  # 96,24,24
        f4 = self.T4(f4)  # 96,12,12
        f4 = self.ODE4(f4)
        temp3, f3 = self.interFA3(f3, f4)
        temp2, f2 = self.interFA2(f2, temp3)
        temp1, f1 = self.interFA1(f1, temp2)
        batch, _, h, w = f1.size()
        f2_temp = F.interpolate(f2, size=(h, w), mode="bilinear", align_corners=True)
        f3_temp = F.interpolate(f3, size=(h, w), mode="bilinear", align_corners=True)
        f4_temp = F.interpolate(f4, size=(h, w), mode="bilinear", align_corners=True)
        feat = torch.cat([f1, f2_temp, f3_temp, f4_temp], 1)
        embed_feat = self.embedding_layer(feat)
        return f1, f2, f3, f4, embed_feat

class GCM_interFA_ODE_slot_full4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM_interFA_ODE_slot_full4, self).__init__()
        self.T1 = ETM(in_channels, out_channels)
        self.T2 = ETM(in_channels * 2, out_channels)
        self.T3 = ETM(in_channels * 4, out_channels)
        self.T4 = ETM(in_channels * 8, out_channels)
        self.ODE1 = ODE(out_channels,num_slots_N=4,num_slots_M=2,iters=3,resolutions=96)
        self.ODE2 = ODE(out_channels,num_slots_N=4,num_slots_M=2,iters=3,resolutions=48)
        self.ODE3 = ODE(out_channels,num_slots_N=4,num_slots_M=2,iters=3,resolutions=24)
        self.ODE4 = ODE(out_channels,num_slots_N=4,num_slots_M=2,iters=3,resolutions=12)
        self.interFA1 = InterFA(out_channels)
        self.interFA2 = InterFA(out_channels)
        self.interFA3 = InterFA(out_channels)

    def forward(self, f1, f2, f3, f4):
        f1 = self.T1(f1)  # 96,96,96
        f1 = self.ODE1(f1)
        f2 = self.T2(f2)  # 96,48,48
        f2 = self.ODE2(f2)
        f3 = self.T3(f3)  # 96,24,24
        f3 = self.ODE3(f3)
        f4 = self.T4(f4)  # 96,12,12
        f4 = self.ODE4(f4)
        temp3, f3 = self.interFA3(f3, f4)
        temp2, f2 = self.interFA2(f2, temp3)
        temp1, f1 = self.interFA1(f1, temp2)

        return f1, f2, f3, f4

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    # paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    # input: B*C*H*W
    # output: B*C*H*W
    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Decoder4(nn.Module):
    def __init__(self, in_channels):
        super(Decoder4, self).__init__()
        self.rcab_a = RCAB(in_channels)
        self.rcab_b = RCAB(in_channels)
        self.crb = BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.out_e = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self.conv3x3 = BasicConv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.out_s = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, f4, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=f4.size()[2:], mode='bilinear', align_corners=True)  # 2,1,12,12->2,1,48,48
        r_prior_cam = 1 - torch.sigmoid(prior_cam)
        prior_cam = torch.sigmoid(prior_cam)
        f4_a = self.rcab_a(f4 * r_prior_cam.expand(-1, f4.size()[1], -1, -1) + f4)
        f4_b = self.rcab_b(f4 * prior_cam.expand(-1, f4.size()[1], -1, -1) + f4)
        f4_s = self.conv3x3(torch.cat([f4_a, f4_b], 1))
        p4_s = self.out_s(f4_s)
        f4_e = self.crb(f4)
        p4_e = self.out_e(f4_e)

        return f4_s, f4_e, p4_s, p4_e


class Spade(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(Spade, self).__init__()
        self.param_free_norm = nn.BatchNorm2d(out_channels, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, edge):
        normalized = self.param_free_norm(x)

        edge = F.interpolate(edge, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(edge)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.rcab_a = RCAB(in_channels)
        self.rcab_b = RCAB(in_channels)
        self.spade_a = Spade(in_channels, in_channels)
        self.spade_b = Spade(in_channels, in_channels)

        self.crb = BasicConv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.out_e = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self.conv3x3 = BasicConv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.out_s = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, f, f_s, f_e, p_s, p_e):
        prior_cam = F.interpolate(p_s, size=f.size()[2:], mode='bilinear', align_corners=True)  # 2,1,12,12->2,1,48,48
        r_prior_cam = 1 - torch.sigmoid(prior_cam)
        prior_cam = torch.sigmoid(prior_cam)

        f_a = self.rcab_a(f * r_prior_cam.expand(-1, f.size()[1], -1, -1) + f)
        f_b = self.rcab_b(f * prior_cam.expand(-1, f.size()[1], -1, -1) + f)
        f_a = self.spade_a(f_a, f_e)
        f_b = self.spade_b(f_b, f_e)

        f_s_new = self.conv3x3(torch.cat([f_a, f_b], 1))
        p_s_new = self.out_s(f_s_new)

        p_e = F.interpolate(p_e, size=f.size()[2:], mode='bilinear', align_corners=True)
        f_s = F.interpolate(f_s, size=f.size()[2:], mode='bilinear', align_corners=True)

        f_e_new = self.crb(torch.cat([(f * p_e.expand(-1, f.size()[1], -1, -1) + f), f_s], 1))
        p_e_new = self.out_e(f_e_new)

        return f_s_new, f_e_new, p_s_new, p_e_new


class REM_decoder(nn.Module):
    def __init__(self, in_channels):
        super(REM_decoder, self).__init__()
        self.decoder4 = Decoder4(in_channels)
        self.decoder3 = Decoder(in_channels)
        self.decoder2 = Decoder(in_channels)
        self.decoder1 = Decoder(in_channels)

    def forward(self, x, prior_cam, pic):
        f1, f2, f3, f4 = x
        f4_s, f4_e, p4_s, p4_e = self.decoder4(f4, prior_cam)
        p4_s_out = F.interpolate(p4_s, size=pic.size()[2:], mode='bilinear')
        p4_e_out = F.interpolate(p4_e, size=pic.size()[2:], mode='bilinear')

        f3_s, f3_e, p3_s, p3_e = self.decoder3(f3, f4_s, f4_e, p4_s, p4_e)
        p3_s_out = F.interpolate(p3_s, size=pic.size()[2:], mode='bilinear')
        p3_e_out = F.interpolate(p3_e, size=pic.size()[2:], mode='bilinear')

        f2_s, f2_e, p2_s, p2_e = self.decoder2(f2, f3_s, f3_e, p3_s, p3_e)
        p2_s_out = F.interpolate(p2_s, size=pic.size()[2:], mode='bilinear')
        p2_e_out = F.interpolate(p2_e, size=pic.size()[2:], mode='bilinear')

        f1_s, f1_e, p1_s, p1_e = self.decoder1(f1, f2_s, f2_e, p2_s, p2_e)
        p1_s_out = F.interpolate(p1_s, size=pic.size()[2:], mode='bilinear')
        p1_e_out = F.interpolate(p1_e, size=pic.size()[2:], mode='bilinear')

        return prior_cam, p4_s_out, p3_s_out, p2_s_out, p1_s_out, p4_e_out, p3_e_out, p2_e_out, p1_e_out


"""
消融实验
"""
class GCM2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM2, self).__init__()
        self.T1 = ETM(in_channels, out_channels)
        self.T2 = ETM(in_channels * 2, out_channels)
        self.T3 = ETM(in_channels * 4, out_channels)
        self.T4 = ETM(in_channels * 8, out_channels)


    def forward(self, f1, f2, f3, f4):
        f1 = self.T1(f1)
        f2 = self.T2(f2)
        f3 = self.T3(f3)
        f4 = self.T4(f4)
        return f1,f2,f3,f4

class Decoder4_noEdge(nn.Module):
    '''
    Input: feature maps from the current stage, the segment map from the previous stage
    Output: the segmentation map from the current stage
    '''
    def __init__(self, in_channels):
        super(Decoder4_noEdge, self).__init__()
        self.rcab_a = RCAB(in_channels)
        self.rcab_b = RCAB(in_channels)

        self.conv3x3 = BasicConv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.out_s = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, f4, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=f4.size()[2:], mode='bilinear', align_corners=True)  # 2,1,12,12->2,1,48,48
        r_prior_cam = 1 - torch.sigmoid(prior_cam)
        prior_cam = torch.sigmoid(prior_cam)
        f4_a = self.rcab_a(f4 * r_prior_cam.expand(-1, f4.size()[1], -1, -1) + f4)
        f4_b = self.rcab_b(f4 * prior_cam.expand(-1, f4.size()[1], -1, -1) + f4)
        f4_s = self.conv3x3(torch.cat([f4_a, f4_b], 1))
        p4_s = self.out_s(f4_s)

        return p4_s


class Decoder_noSpade(nn.Module):
    def __init__(self, in_channels):
        super(Decoder_noSpade, self).__init__()
        self.rcab_a = RCAB(in_channels)
        self.rcab_b = RCAB(in_channels)

        self.crb = BasicConv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.out_e = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self.conv3x3 = BasicConv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.out_s = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, f, f_s, f_e, p_s, p_e):
        prior_cam = F.interpolate(p_s, size=f.size()[2:], mode='bilinear', align_corners=True)  # 2,1,12,12->2,1,48,48
        r_prior_cam = 1 - torch.sigmoid(prior_cam)
        prior_cam = torch.sigmoid(prior_cam)

        f_a = self.rcab_a(f * r_prior_cam.expand(-1, f.size()[1], -1, -1) + f)
        f_b = self.rcab_b(f * prior_cam.expand(-1, f.size()[1], -1, -1) + f)

        f_s_new = self.conv3x3(torch.cat([f_a, f_b], 1))
        p_s_new = self.out_s(f_s_new)

        p_e = F.interpolate(p_e, size=f.size()[2:], mode='bilinear', align_corners=True)
        f_s = F.interpolate(f_s, size=f.size()[2:], mode='bilinear', align_corners=True)

        f_e_new = self.crb(torch.cat([(f * p_e.expand(-1, f.size()[1], -1, -1) + f), f_s], 1))
        p_e_new = self.out_e(f_e_new)

        return f_s_new, f_e_new, p_s_new, p_e_new

class REM_decoder_noSpade(nn.Module):
    def __init__(self, in_channels):
        super(REM_decoder_noSpade, self).__init__()
        self.decoder4 = Decoder4(in_channels)
        self.decoder3 = Decoder_noSpade(in_channels)
        self.decoder2 = Decoder_noSpade(in_channels)
        self.decoder1 = Decoder_noSpade(in_channels)

    def forward(self, x, prior_cam, pic):
        f1, f2, f3, f4 = x
        f4_s, f4_e, p4_s, p4_e = self.decoder4(f4, prior_cam)
        p4_s_out = F.interpolate(p4_s, size=pic.size()[2:], mode='bilinear')
        p4_e_out = F.interpolate(p4_e, size=pic.size()[2:], mode='bilinear')

        f3_s, f3_e, p3_s, p3_e = self.decoder3(f3, f4_s, f4_e, p4_s, p4_e)
        p3_s_out = F.interpolate(p3_s, size=pic.size()[2:], mode='bilinear')
        p3_e_out = F.interpolate(p3_e, size=pic.size()[2:], mode='bilinear')

        f2_s, f2_e, p2_s, p2_e = self.decoder2(f2, f3_s, f3_e, p3_s, p3_e)
        p2_s_out = F.interpolate(p2_s, size=pic.size()[2:], mode='bilinear')
        p2_e_out = F.interpolate(p2_e, size=pic.size()[2:], mode='bilinear')

        f1_s, f1_e, p1_s, p1_e = self.decoder1(f1, f2_s, f2_e, p2_s, p2_e)
        p1_s_out = F.interpolate(p1_s, size=pic.size()[2:], mode='bilinear')
        p1_e_out = F.interpolate(p1_e, size=pic.size()[2:], mode='bilinear')

        return prior_cam, p4_s_out, p3_s_out, p2_s_out, p1_s_out, p4_e_out, p3_e_out, p2_e_out, p1_e_out

class REM_decoder_noSpade_noEdge(nn.Module):
    def __init__(self, in_channels):
        super(REM_decoder_noSpade_noEdge, self).__init__()
        self.decoder4 = Decoder4_noEdge(in_channels)#可以弄4个decoder4
        self.decoder3 = Decoder4_noEdge(in_channels)
        self.decoder2 = Decoder4_noEdge(in_channels)
        self.decoder1 = Decoder4_noEdge(in_channels)

    def forward(self, x, prior_cam, pic):
        f1, f2, f3, f4 = x
        # 把所有的edge去掉。
        p4_s = self.decoder4(f4, prior_cam)
        p4_s_out = F.interpolate(p4_s, size=pic.size()[2:], mode='bilinear')

        p3_s = self.decoder3(f3, p4_s)
        p3_s_out = F.interpolate(p3_s, size=pic.size()[2:], mode='bilinear')

        p2_s = self.decoder2(f2, p3_s)
        p2_s_out = F.interpolate(p2_s, size=pic.size()[2:], mode='bilinear')
        # p2_e_out = F.interpolate(p2_e, size=pic.size()[2:], mode='bilinear')

        p1_s = self.decoder1(f1, p2_s)
        p1_s_out = F.interpolate(p1_s, size=pic.size()[2:], mode='bilinear')
        # p1_e_out = F.interpolate(p1_e, size=pic.size()[2:], mode='bilinear')

        return prior_cam, p4_s_out, p3_s_out, p2_s_out, p1_s_out

class Decoder4_noRCAB(nn.Module):
    def __init__(self, in_channels):
        super(Decoder4_noRCAB, self).__init__()
        self.crb = BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.out_e = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self.conv3x3 = BasicConv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.out_s = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, f4, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=f4.size()[2:], mode='bilinear', align_corners=True)  # 2,1,12,12->2,1,48,48
        r_prior_cam = 1 - torch.sigmoid(prior_cam)
        prior_cam = torch.sigmoid(prior_cam)
        f4_a = f4 * r_prior_cam.expand(-1, f4.size()[1], -1, -1) + f4
        f4_b = f4 * prior_cam.expand(-1, f4.size()[1], -1, -1) + f4
        f4_s = self.conv3x3(torch.cat([f4_a, f4_b], 1))
        p4_s = self.out_s(f4_s)
        f4_e = self.crb(f4)
        p4_e = self.out_e(f4_e)

        return f4_s, f4_e, p4_s, p4_e

class Decoder_noRCAB(nn.Module):
    def __init__(self, in_channels):
        super(Decoder_noRCAB, self).__init__()

        self.spade_a = Spade(in_channels, in_channels)
        self.spade_b = Spade(in_channels, in_channels)

        self.crb = BasicConv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.out_e = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self.conv3x3 = BasicConv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.out_s = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, f, f_s, f_e, p_s, p_e):
        prior_cam = F.interpolate(p_s, size=f.size()[2:], mode='bilinear', align_corners=True)  # 2,1,12,12->2,1,48,48
        r_prior_cam = 1 - torch.sigmoid(prior_cam)
        prior_cam = torch.sigmoid(prior_cam)

        f_a = f * r_prior_cam.expand(-1, f.size()[1], -1, -1) + f
        f_b = f * prior_cam.expand(-1, f.size()[1], -1, -1) + f
        f_a = self.spade_a(f_a, f_e)
        f_b = self.spade_b(f_b, f_e)

        f_s_new = self.conv3x3(torch.cat([f_a, f_b], 1))
        p_s_new = self.out_s(f_s_new)

        p_e = F.interpolate(p_e, size=f.size()[2:], mode='bilinear', align_corners=True)
        f_s = F.interpolate(f_s, size=f.size()[2:], mode='bilinear', align_corners=True)

        f_e_new = self.crb(torch.cat([(f * p_e.expand(-1, f.size()[1], -1, -1) + f), f_s], 1))
        p_e_new = self.out_e(f_e_new)

        return f_s_new, f_e_new, p_s_new, p_e_new

class REM_decoder_noRCAB(nn.Module):
    def __init__(self, in_channels):
        super(REM_decoder_noRCAB, self).__init__()
        self.decoder4 = Decoder4_noRCAB(in_channels)
        self.decoder3 = Decoder_noRCAB(in_channels)
        self.decoder2 = Decoder_noRCAB(in_channels)
        self.decoder1 = Decoder_noRCAB(in_channels)

    def forward(self, x, prior_cam, pic):
        f1, f2, f3, f4 = x
        f4_s, f4_e, p4_s, p4_e = self.decoder4(f4, prior_cam)
        p4_s_out = F.interpolate(p4_s, size=pic.size()[2:], mode='bilinear')
        p4_e_out = F.interpolate(p4_e, size=pic.size()[2:], mode='bilinear')

        f3_s, f3_e, p3_s, p3_e = self.decoder3(f3, f4_s, f4_e, p4_s, p4_e)
        p3_s_out = F.interpolate(p3_s, size=pic.size()[2:], mode='bilinear')
        p3_e_out = F.interpolate(p3_e, size=pic.size()[2:], mode='bilinear')

        f2_s, f2_e, p2_s, p2_e = self.decoder2(f2, f3_s, f3_e, p3_s, p3_e)
        p2_s_out = F.interpolate(p2_s, size=pic.size()[2:], mode='bilinear')
        p2_e_out = F.interpolate(p2_e, size=pic.size()[2:], mode='bilinear')

        f1_s, f1_e, p1_s, p1_e = self.decoder1(f1, f2_s, f2_e, p2_s, p2_e)
        p1_s_out = F.interpolate(p1_s, size=pic.size()[2:], mode='bilinear')
        p1_e_out = F.interpolate(p1_e, size=pic.size()[2:], mode='bilinear')

        return prior_cam, p4_s_out, p3_s_out, p2_s_out, p1_s_out, p4_e_out, p3_e_out, p2_e_out, p1_e_out


if __name__ == '__main__':
    # f3 = torch.randn(2, 64, 96, 96).cuda()
    # net = SlotAttentionModule2(encoder_dims=64, resolution=96, num_slots =4, iters =3).cuda()
    # y = net(f3)
    # print(y[0].shape)
    # print(y[1].shape)
    f0 = torch.randn(2, 256, 96, 96).cuda()
    f1 = torch.randn(2, 512, 48, 48).cuda()
    f2 = torch.randn(2, 1024, 24, 24).cuda()
    f3 = torch.randn(2, 2048, 12, 12).cuda()
    gcm4 = GCM_interFA_ODE_slot(256, 64).cuda()
    f1, f2, f3, f4 = gcm4(f0, f1, f2, f3)
    # print(f1.shape)
    # print(f2.shape)
    # print(f3.shape)
    # print(f4.shape)
    # f0 = torch.randn(2, 1, 12, 12).cuda()
    # pict = torch.randn(2, 3, 384, 384).cuda()
    # rem = REM_decoder(96).cuda()
    # prior_cam, p4_s_out, p3_s_out, p2_s_out, p1_s_out, p4_e_out, p3_e_out, p2_e_out, p1_e_out = rem([f1, f2, f3, f4],f0,pict)

    # print(prior_cam.shape)
    # print(p4_s_out.shape)
    # print(p3_s_out.shape)
    # print(p2_s_out.shape)
    # print(p1_s_out.shape)
    # print(p4_e_out.shape)
    # print(p3_e_out.shape)
    # print(p2_e_out.shape)
    # print(p1_e_out.shape)


    # f0 = torch.randn(2, 1, 12, 12).cuda()
    # f4 = torch.randn(2, 96, 12, 12).cuda()
    # f3 = torch.randn(2, 96, 24, 24).cuda()
    # f2 = torch.randn(2, 96, 48, 48).cuda()
    # f1 = torch.randn(2, 96, 96, 96).cuda()
    # pict = torch.randn(2, 3, 384, 384).cuda()
    # x = [f1, f2, f3, f4]
    # sr = SR(96, 96).cuda()
    # f4, f3, f2, f1, bound_f1 = sr(x, f0, pict)
    # print(f4.shape)
    # print(f3.shape)
    # print(f2.shape)
    # print(f1.shape)
    # print(bound_f1.shape)
    # f1 = torch.randn(2, 1, 12, 12).cuda()
    # ll = torch.randn(2, 64, 96, 96).cuda()
    # lh = torch.randn(2, 64, 48, 48).cuda()
    # hl = torch.randn(2, 64, 24, 24).cuda()
    # hh = torch.randn(2, 64, 12, 12).cuda()
    # pict = torch.randn(2, 3, 384, 384).cuda()
    # x = [ll, lh, hl, hh]
    # rem = REM12(64,64).cuda()
    # f4,f3,f2,f1,bound_f4,bound_f3,bound_f2,bound_f1 = rem(x, f1, pict)
    # print(f4.shape)
    # print(f3.shape)
    # print(f2.shape)
    # print(f1.shape)
    # print(bound_f4.shape)
    # print(bound_f3.shape)
    # print(bound_f2.shape)
    # print(bound_f1.shape)

    # input = torch.randn(2, 2048, 16, 16)
    # model = GPM(depth=128)
    # total = sum([param.nelement() for param in model.parameters()])
    # print('Number of parameter: %.2fM' % (total/1e6))
    # flops, params = profile(model, inputs=(input, ))
    # print('flops:{}'.format(flops*2))
    # print('params:{}'.format(params))
