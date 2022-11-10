import torch.nn as nn
from typing import Optional, Union, List
from .model_config import MODEL_CONFIG
from .decoder.sanet import SANetDecoder
from .get_encoder import build_encoder
from .base_model import SegmentationModel


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class SANet(SegmentationModel):
    """SANet is a fully convolution neural network for image semantic segmentation. Consist of *encoder* 
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial 
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.
    Args:
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_channels: List of integers which specify **out_channels** parameter for convolutions used in encoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNormalization layer between Conv2D and Activation layers is used.
            Available options are **True, False**.
        decoder_attention_type: Attention module used in decoder of the model. Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        upsampling: Int number of upsampling factor for segmentation head, default=1 
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        aux_classifier: If **True**, add a classification branch based the last feature of the encoder.
            Available options are **True, False**.
    Returns:
        ``torch.nn.Module``: SANet
    """

    def __init__(
        self,
        in_channels: int = 3,
        encoder_name: str = "simplenet",
        encoder_weights: Optional[str] = None,
        encoder_depth: int = 5,
        encoder_channels: List[int] = [32,64,128,256,512],
        encoder_output_stride: int = 32,
        num_stage: int = 4,
        decoder_use_batchnorm: bool = True,
        decoder_attention_type: Optional[str] = None,
        decoder_channels: List[int] = [128],
        upsampling: int = 1,
        classes: int = 1,
        aux_classifier: bool = False,
    ):
        super().__init__()

        self.encoder_depth = encoder_depth
        self.encoder_channels = encoder_channels

        self.encoder = build_encoder(
            encoder_name,
            weights=encoder_weights,
            n_channels=in_channels
        )


        if encoder_output_stride == 8:
            self.make_dilated(
                stage_list=[3, 4],
                dilation_list=[2, 4]
            )

        elif encoder_output_stride == 16:
            self.make_dilated(
                stage_list=[4],
                dilation_list=[2]
            )
        elif encoder_output_stride == 32:
            pass
        else:
            raise ValueError(
                "Encoder output stride should be 8 or 16 or 32, got {}".format(encoder_output_stride)
            )

        self.decoder = SANetDecoder(
            encoder_channels=self.encoder_channels,
            num_stage=num_stage,
            decoder_channels=decoder_channels, 
            use_batchnorm=decoder_use_batchnorm,
            attention_type=decoder_attention_type,
            norm_layer=nn.BatchNorm2d
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity(),
        )

        if aux_classifier:
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(self.encoder_channels[-1], classes - 1, bias=True)
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()


    def make_dilated(self, stage_list, dilation_list):
        stages = self.encoder.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            self.replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )


    def replace_strides_with_dilation(self, module, dilation_rate):
        """Patch Conv2d modules replacing strides with dilation"""
        for mod in module.modules():
            if isinstance(mod, nn.Conv2d):
                mod.stride = (1, 1)
                mod.dilation = (dilation_rate, dilation_rate)
                kh, kw = mod.kernel_size
                mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

                # Kostyl for EfficientNet
                if hasattr(mod, "static_padding"):
                    mod.static_padding = nn.Identity()


def sanet(model_name,encoder_name,**kwargs):
    params = MODEL_CONFIG[model_name][encoder_name]
    dynamic_params = kwargs
    for key in dynamic_params:
        if key in params:
            params[key] = dynamic_params[key]

    net = SANet(**params)
    return net
