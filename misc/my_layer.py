import torch
import torch.nn as nn
from misc.layer import Conv2d,MobileNet
import torch.nn.functional as F
from misc import common
from misc.tool import extract_image_patches,reduce_mean, reduce_sum, same_padding


class PyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()
        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)
        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                               stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                               dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))
        return torch.cat(out, 1)

class PyDConv2d(nn.Module):
    def __init__(self, in_channels, channels, out_channels, pyconv_kernels, pyconv_dilation, pyconv_groups, stride=1, bias=False):
        super(PyDConv2d, self).__init__()
        assert len(channels) == len(pyconv_dilation) == len(pyconv_groups)
        self.pyconv_levels = [None] * len(pyconv_dilation)
        for i in range(len(pyconv_dilation)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, channels[i], kernel_size=pyconv_kernels,
                            stride=stride, padding=pyconv_dilation[i], groups=pyconv_groups[i],
                            dilation=pyconv_dilation[i], bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)
        self.Pybn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))
        out = torch.cat(out, 1)
        out = self.Pybn(out)
        out = self.relu(out)
        return out

class light_PyDConv2d(nn.Module):
    def __init__(self, in_channels, channels, out_channels, pyconv_dilation, pyconv_groups, stride=1, bias=False):
        super(light_PyDConv2d, self).__init__()
        assert len(channels) == len(pyconv_dilation) == len(pyconv_groups)
        self.pyconv_levels = [None] * len(pyconv_dilation)
        for i in range(len(pyconv_dilation)):
            self.pyconv_levels[i]=MobileNet(in_channels,[channels[i]],dilation=pyconv_dilation[i])
            # self.pyconv_levels[i] = nn.Conv2d(in_channels, channels[i], kernel_size=3,
            #                 stride=stride, padding=pyconv_dilation[i], groups=pyconv_groups[i],
            #                 dilation=pyconv_dilation[i], bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)
        self.Pybn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            # print('level:',level(x).shape)
            out.append(level(x))
            # print('each pyconv:',out)
        out = torch.cat(out, 1)
        # print('cat:',out.shape)
        out = self.Pybn(out)
        # print('pybn:',out.shape)
        out = self.relu(out)
        # print('relu:',out.shape)
        return out

"""ttention modules"""

class SELayer(nn.Module):
    def __init__(self,channel,reduction=16):
        super(SELayer,self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,h,w=x.size()
        y=self.avgpool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)

class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
    def forward(self, U):
        q = self.Conv1x1(U)
        q = self.norm(q)
        return U * q # 广播机制
class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2,kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels,kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)
class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)
    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse


class ResidualBlock(nn.Module):
    def __init__(self,in_channels, channels, out_channels, pyconv_dilation,pyconv_groups,stride=1, bias=False):
        super (ResidualBlock,self).__init__()
        self.conv1=Conv2d(in_channels,in_channels,1,stride=1,NL='relu',bn=True,bias=False)
        self.pyDconv= light_PyDConv2d(in_channels, channels, out_channels, pyconv_dilation, pyconv_groups,stride=1, bias=False)
        # self.scSE = scSE(out_channels)
        self.conv2=Conv2d(out_channels,out_channels,1,stride=1,NL='relu',bn=True,bias=False)
        # self.scSE=scSE(out_channels)
        if in_channels==out_channels:
            self.shortcut=nn.Sequential(
                # SELayer(out_channels)
                # scSE(out_channels)
            )
        else:
            self.shortcut=nn.Sequential(
                Conv2d(in_channels,out_channels,1,stride=1,NL='relu',bn=True,bias=False),
                # SELayer(out_channels)
                scSE(out_channels)

            )
    def forward(self,x):
        residual=x
        x1=self.conv1(x)
        x2=self.pyDconv(x1)
        # x2=self.scSE(x2)
        x3=self.conv2(x2)
        # x3=self.scSE(x3)
        x4=x3+self.shortcut(residual)
        return x4

class ResidualBlock2(nn.Module):
    def __init__(self,in_channels, channels, out_channels, pyconv_dilation,pyconv_groups,stride=1, bias=False):
        super (ResidualBlock,self).__init__()
        self.conv1=Conv2d(in_channels,in_channels,1,stride=1,NL='relu',bn=True,bias=False)
        self.pyDconv= light_PyDConv2d(in_channels, channels, out_channels, pyconv_dilation, pyconv_groups,stride=1, bias=False)
        # self.scSE = scSE(out_channels)
        self.conv2=Conv2d(out_channels,out_channels,1,stride=1,NL='relu',bn=True,bias=False)
        self.scSE=scSE(out_channels)
        if in_channels==out_channels:
            self.shortcut=nn.Sequential(
                # SELayer(out_channels)
                # scSE(out_channels)
            )
        else:
            self.shortcut=nn.Sequential(
                Conv2d(in_channels,out_channels,1,stride=1,NL='relu',bn=True,bias=False),
                # SELayer(out_channels)
                scSE(out_channels)

            )
        self.shortcut2=nn.Sequential(
            Conv2d(in_channels, out_channels, 1, stride=1, NL='relu', bn=True, bias=False),
        )

    def myforward(self,x,y):
        residual=x
        x1=self.conv1(x)
        x2=self.pyDconv(x1)
        # x2=self.scSE(x2)
        x3=self.conv2(x2)
        x3=self.scSE(x3)
        x4=x3+self.shortcut(residual)+self.shortcut2(y)
        return x4



class ExBlock(nn.Module):
    def __init__(self, high_in_plane, low_in_plane, out_plane):
        super(ExBlock, self).__init__()
        self.convhigh = nn.Conv2d(high_in_plane, out_plane, 1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.convlow = nn.Conv2d(low_in_plane, out_plane, 1)

    def forward(self, high_x, low_x):
        high_x = self.upsample(self.convhigh(high_x))
        low_x = self.convlow(low_x)
        return high_x + low_x

class EFFBlock(nn.Module):
    def __init__(self, high_in_plane, low_in_plane, out_plane):
        super(EFFBlock, self).__init__()
        self.convhigh = nn.Conv2d(high_in_plane, out_plane, 1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.convlow = ASPPblock(low_in_plane,out_plane)

    def forward(self, high_x, low_x):
        high_x = self.upsample(self.convhigh(high_x))
        low_x = self.convlow(low_x)
        return high_x * low_x

class MSAModule(nn.Module):
    def __init__(self, in_channel):
        super(MSAModule, self).__init__()
        self.branch1_conv3x3 = nn.Conv2d(in_channel, in_channel // 2, 3, 1, padding=1, dilation=1)
        self.branch2_conv3x3_1 = nn.Conv2d(in_channel, in_channel // 2, 3, 1, padding=1, dilation=1)
        self.branch2_conv3x3_2 = nn.Conv2d(in_channel // 2, in_channel // 2, 3, 1, padding=1, dilation=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x1 = self.relu(self.branch1_conv3x3(x))
        x2 = self.relu(self.branch2_conv3x3_1(x))
        x2 = self.relu(self.branch2_conv3x3_2(x2))
        return x1 + x2

class ASPPblock(nn.Module):
    def __init__(self, in_channel=512,out_chanel=512):
        super(ASPPblock, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channel, in_channel // 4, 3, 1, padding=1, dilation=1)
        self.atrous_block2 = nn.Conv2d(in_channel, in_channel // 4, 3, 1, padding=2, dilation=2)
        self.atrous_block3 = nn.Conv2d(in_channel, in_channel // 4, 3, 1, padding=3, dilation=3)
        self.atrous_block4 = nn.Conv2d(in_channel, in_channel // 4, 3, 1, padding=4, dilation=4)
        self.conv_output = nn.Conv2d(in_channel, out_chanel, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        atrous_block1 = self.relu(self.atrous_block1(x))
        atrous_block2 = self.relu(self.atrous_block2(x))
        atrous_block3 = self.relu(self.atrous_block3(x))
        atrous_block4 = self.relu(self.atrous_block4(x))

        net = torch.cat([atrous_block1, atrous_block2, atrous_block3, atrous_block4], dim=1)
        net = self.relu(self.conv_output(net))
        return net

class CEMBlock(nn.Module):
    def __init__(self, inn, out, rate=4):
        super(CEMBlock, self).__init__()
        base = inn // rate
        self.conv_sa = nn.Sequential(Conv2d(inn, base, 3, same_padding=True, bn=True, bias=False),
                        ContextBlock(base, ratio=0.25, pooling_type='att'),
                        Conv2d(base, base, 3, same_padding=True, bn=True, bias=False)
                       )
        self.conv_ca = nn.Sequential(Conv2d(inn, base, 3, same_padding=True, bn=True, bias=False),
                       ContextBlock(base, ratio=0.25, pooling_type='avg'),
                       Conv2d(base, base, 3, same_padding=True, bn=True, bias=False)
                       )
        self.conv_cat = Conv2d(base*2, out, 1, same_padding=True, bn=False)


    def forward(self, x):
        sa_feat = self.conv_sa(x)
        ca_feat = self.conv_ca(x)
        cat_feat = torch.cat((sa_feat, ca_feat), 1)
        cat_feat = self.conv_cat(cat_feat)
        return cat_feat

class ContextBlock(nn.Module):
    def __init__(self, inplanes, ratio=0.25, pooling_type='att'):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

    def forward(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        channel_add_term = self.channel_add_conv(context)
        out = x + channel_add_term
        return out

class SEblock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ContextBlock_GCNet(nn.Module):
    def __init__(self,inplanes,ratio,pooling_type='att',fusion_types=('channel_add', )):
        super(ContextBlock_GCNet, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True), # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True), # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()

        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):

        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
        # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out

class ContextBlock_CA(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(ContextBlock_CA, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features, features, kernel_size=1)

    def __make_weight(self, feature, scale_feature):
        weight_feature = feature - scale_feature
        return F.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        multi_scales = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.scales]
        # print(multi_scales)
        weights = [self.__make_weight(feats, scale_feature) for scale_feature in multi_scales]
        overall_features = [(multi_scales[0] * weights[0] + multi_scales[1] * weights[1] + multi_scales[2] * weights[
            2] + multi_scales[3] * weights[3]) / (weights[0] + weights[1] + weights[2] + weights[3])] + [feats]
        bottle = self.bottleneck(torch.cat(overall_features, 1))
        return self.relu(bottle)

class ASD(nn.Module):
    def __init__(self, channel,reduction):
        super(ASD, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y.expand_as(x)

class ghostnet_attention(nn.Module):
    def __init__(self,oup,new_channel , dilation,dw_size=3,relu=True):
        super(ghostnet_attention, self).__init__()
        self.cheap_operation = nn.Sequential(
                    # MobileNet(oup, [new_channel], dilation=dilation, kernel=3),
                    nn.Conv2d(oup,new_channel , dw_size, 1, padding=dilation,groups=new_channel,dilation=dilation,bias=False),
                    nn.BatchNorm2d(new_channel),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
    def forward(self,x):
        x=self.cheap_operation(x)
        return x

class PyramidAttention(nn.Module):
    def __init__(self, oup,new_channel, pyconv_dilation,level=5, res_scale=5, channel=336, reduction=2, ksize=3, stride=1, softmax_scale=10, average=True,
                 conv=common.default_conv):
        super(PyramidAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.scale = [1]
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_L_base = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())
        self.ghostnet=ghostnet_attention(oup,new_channel, pyconv_dilation)

    def forward(self, input):
        res = input
        # theta
        match_base = self.conv_match_L_base(input)
        # print('mch_base:',match_base.shape)
        shape_base = list(res.size())
        input_groups = torch.split(match_base, 1, dim=0)

        # patch size for matching
        kernel = self.ksize
        # raw_w is for reconstruction
        raw_w = []
        # w is for matching
        w = []
        # build feature pyramid
        """
        for i in range(len(self.scale)):
            ref = input
            if self.scale[i] != 1:
                ref = F.interpolate(input, scale_factor=self.scale[i], mode='bicubic')
                # print('ref:',ref.shape,'self.scale:',self.scale[i])
        """
        ref=self.ghostnet(input)

        # feature transformation function f
        base = self.conv_assembly(ref)
        # print('base:',base.shape)

        shape_input = base.shape
        # sampling
        raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
                                        strides=[self.stride, self.stride],
                                        rates=[1, 1],
                                        padding='same')  # [N, C*k*k, L]
        raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], kernel, kernel, -1)
        raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
        raw_w.append(raw_w_i_groups)

        # feature transformation function g
        ref_i = self.conv_match(ref)
        shape_ref = ref_i.shape
        # sampling
        w_i = extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize],
                                    strides=[self.stride, self.stride],
                                    rates=[1, 1],
                                    padding='same')
        w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
        w_i = w_i.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
        w_i_groups = torch.split(w_i, 1, dim=0)
        w.append(w_i_groups)
        # print('w:',w.shape)



        y = []
        for idx, xi in enumerate(input_groups):
            # group in a filter
            wi = torch.cat([w[i][idx][0] for i in range(len(self.scale))], dim=0)  # [L, C, k, k]
            # normalize
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi / max_wi
            # matching
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            # print('xi',xi.shape)
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
            # print('yi_conv2d',yi.shape)
            yi = yi.view(1, wi.shape[0], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)
            # print('view:',yi.shape)

            # softmax matching score
            yi = F.softmax(yi * self.softmax_scale, dim=1)

            if self.average == False:
                yi = (yi == yi.max(dim=1, keepdim=True)[0]).float()

            # deconv for patch pasting
            raw_wi = torch.cat([raw_w[i][idx][0] for i in range(len(self.scale))], dim=0)
            yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride, padding=1) / 4.
            y.append(yi)

        y = torch.cat(y, dim=0) + res * self.res_scale  # back to the mini-batch
        return y