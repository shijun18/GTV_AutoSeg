import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module import attention as attention



class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
            norm_layer=None,
            activation=True
    ):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = norm_layer
        

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = norm_layer(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu if activation else nn.Identity())




class DilatedBlock(nn.Sequential):

    def __init__(self,in_channel,out_channel,dilation=1, norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = norm_layer

        conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        
        bn = norm_layer(out_channel)

        relu = nn.ReLU(inplace=True)

        super(DilatedBlock, self).__init__(conv, bn, relu)



class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels) if norm_layer is not None else nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[8,16,32], norm_layer=None):
        super(ASPP, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = norm_layer

        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = atrous_rates

        modules.append(DilatedBlock(in_channels, out_channels, dilation=rate1, norm_layer=norm_layer))
        modules.append(DilatedBlock(in_channels, out_channels, dilation=rate2, norm_layer=norm_layer))
        modules.append(DilatedBlock(in_channels, out_channels, dilation=rate3, norm_layer=norm_layer))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_chans, out_chans, sizes=(1, 2, 3, 6), norm_layer=None):
        super(PSPModule, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        len_size = len(sizes)
        self.stages = []
        self.stages = nn.ModuleList(
                [self._make_stage(in_chans, out_chans, size, norm_layer) for size in sizes]
            )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(int(in_chans + len_size*out_chans), out_chans, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_chans),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle




class DynamicAlignModulev2(nn.Module):
    def __init__(self, inplane, outplane):
        super(DynamicAlignModulev2, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Sequential(
            nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )
        
        self.grid = nn.Parameter(torch.zeros(1, 1, 1, 2))

    def forward(self, low_feature, h_feature):
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)

        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.interpolate(h_feature,size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, _, _, _ = input.size()
        grid = self.grid.repeat(n, out_h, out_w, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1)
        output = F.grid_sample(input, grid)
        return output



class DynamicAlignModule(nn.Module):
    def __init__(self, inplane, outplane):
        super(DynamicAlignModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False)
        self.grid = nn.Parameter(torch.zeros(1, 1, 1, 2))

    def forward(self, low_feature, h_feature):
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)

        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.interpolate(h_feature,size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, _, _, _ = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        grid = self.grid.repeat(n, out_h, out_w, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class NoAlignModule(nn.Module):
    def __init__(self, inplane, outplane):
        super(NoAlignModule, self).__init__()
       
    def forward(self, low_feature, h_feature):
        h, w = low_feature.size()[2:]
        size = (h, w)
        h_feature = F.interpolate(h_feature,size=size, mode="bilinear", align_corners=False)

        return h_feature



class AlignModule(nn.Module):
    def __init__(self, inplane, outplane):
        super(AlignModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, low_feature, h_feature):
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)

        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.interpolate(h_feature,size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class AlignHead(nn.Module):
    def __init__(self, in_chans, fpn_inplanes, fpn_dim=256, norm_layer=None, align_module=None, bottleneck=None, attention_type=None):
        super(AlignHead, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d


        self.bottleneck = bottleneck(in_chans, fpn_dim,norm_layer=norm_layer) if bottleneck is not None else nn.Identity()
        self.fpn_in = nn.ModuleList()
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.fpn_out = nn.ModuleList()
        self.fpn_out_align = nn.ModuleList()
        self.attention_out = nn.ModuleList()
        for _ in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(
                Conv2dReLU(fpn_dim, fpn_dim, 3, stride=1, padding=1,use_batchnorm=True,norm_layer=norm_layer)
            )
            self.fpn_out_align.append(
                align_module(inplane=fpn_dim, outplane=fpn_dim//2)
            )
            self.attention_out.append(
                attention.Attention(attention_type, in_channels=fpn_dim)
            )

    def forward(self, conv_out):
        bottleneck = self.bottleneck(conv_out[-1])

        f = bottleneck
        fpn_feature_list = [bottleneck]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch
            f = self.fpn_out_align[i](conv_x, f)
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](self.attention_out[i](f)))
            out.append(f)

        fpn_feature_list.reverse() 
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))

        fusion_out = torch.cat(fusion_list, 1)
        return fusion_out, out




class SANetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            num_stage,
            decoder_channels,
            use_batchnorm=True,
            attention_type=None,
            norm_layer=None,
    ):
        super().__init__()
        # computing blocks input and output channels
        assert len(decoder_channels) == 1
        assert num_stage <= len(encoder_channels)
        self.off_set = len(encoder_channels) - num_stage
        num_features = encoder_channels[-1]

        # fpn_dim = max(num_features // 8, 128)
        fpn_dim = decoder_channels[0]
        self.head = AlignHead(num_features, 
                    fpn_inplanes=encoder_channels[self.off_set:],
                    fpn_dim=fpn_dim,
                    norm_layer=norm_layer,
                    align_module=AlignModule,
                    bottleneck=PSPModule, #PSPModule, ASPP
                    attention_type=attention_type
                )
        self.conv_last = Conv2dReLU(int(num_stage * fpn_dim), fpn_dim, 3, stride=1, padding=1,use_batchnorm=use_batchnorm,norm_layer=norm_layer)


    def forward(self, *features):

        x_ = features[self.off_set:]                                                                                                
        x, _ = self.head(x_)
        x = self.conv_last(x)
        return x