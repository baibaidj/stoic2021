# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmcv.runner.base_module import BaseModule
from mmcv.utils import to_3tuple
import pdb


def nlc_to_nchwd(x, hwd_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W, D] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hwd_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W, D = hwd_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W * D, 'The seq_len does not match H, W, D'
    return x.transpose(1, 2).reshape(B, C, H, W, D).contiguous()


def nchwd_to_nlc(x):
    """Flatten [N, C, H, W, D] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W, D] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 5
    return x.flatten(2).transpose(1, 2).contiguous()


class AdaptivePadding3D(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):

        super(AdaptivePadding3D, self).__init__()

        assert padding in ('same', 'corner')

        kernel_size = to_3tuple(kernel_size)
        stride = to_3tuple(stride)
        padding = to_3tuple(padding)
        dilation = to_3tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w, input_d = input_shape
        kernel_h, kernel_w, kernel_d = self.kernel_size
        stride_h, stride_w, stride_d = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        output_d = math.ceil(input_d / stride_d)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        pad_d = max((output_d - 1) * stride_d + (kernel_d - 1) * self.dilation[2] + 1 - input_d, 0)

        return pad_h, pad_w, pad_d

    def forward(self, x):
        pad_h, pad_w, pad_d = self.get_pad_shape(x.size()[2:])
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h, 0, pad_d])
            elif self.padding == 'same':
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, 
                             pad_h // 2, pad_h - pad_h // 2, 
                            pad_d // 2, pad_d - pad_d // 2, 
                    ])
        return x


class PatchEmbed3D(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        in_channels=1,
        embed_dims=32,
        conv_type='Conv3d',
        kernel_size=16,
        stride=16,
        padding='corner',
        dilation=1,
        bias=True,
        norm_cfg=None,
        input_size=None,
        init_cfg=None,
    ):
        super(PatchEmbed3D, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_3tuple(kernel_size)
        stride = to_3tuple(stride)
        dilation = to_3tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding3D(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        
        # pdb.set_trace()
        padding = to_3tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_3tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w, pad_d = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w, input_d = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_d = input_d + pad_d
                input_size = (input_h, input_w, input_d)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            d_out = (input_size[2] + 2 * padding[2] - dilation[2] *
                     (kernel_size[2] - 1) - 1) // stride[2] + 1
            self.init_out_size = (h_out, w_out, d_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W, D). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w * out_d, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w, out_d).
        """

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = x.shape[2:]
        x = x.flatten(2).transpose(1, 2) # BCN > BNC
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class PatchMerging3D(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, H, W, D):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W*D, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W * D, "input feature has wrong size"

        x = x.view(B, H, W, D, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (D % 2== 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, D % 2 , 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 D/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 D/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 D/2 C
        x3 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 D/2 C

        x4 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 D/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 D/2 C
        x6 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 D/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 D/2 C

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 D/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*D/2  8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed3DNN(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        in_channels=1,
        embed_dims=32,
        conv_type='Conv3d',
        kernel_size=3,
        stride=4,
        padding='corner',
        dilation=1,
        bias=True,
        norm_cfg= dict(type = 'LN'),
        input_size=None,
        init_cfg=None,
    ):
        super(PatchEmbed3DNN, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_3tuple(kernel_size)
        stride = to_3tuple(stride)
        dilation = to_3tuple(dilation)

        print('[PatchEmbed] kernel', kernel_size)
        print('[PatchEmbed] stride', stride)
        print('[PatchEmbed] dilation', dilation)


        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding3D(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        
        # pdb.set_trace()
        padding = to_3tuple(padding)

        self.projection1 = ConvLnGeluSequence(in_channels, 
                                            embed_dims//2,
                                            kernel_size=kernel_size, 
                                            stride = [a//2 for a in stride], 
                                            padding = [(a-1)//2 for a in kernel_size], 
                                            norm_cfg= norm_cfg, 
                                            conv_type=conv_type, 
                                            bias = bias)

        self.projection2 = ConvLnGeluSequence(embed_dims//2, 
                                            embed_dims,
                                            kernel_size=kernel_size, 
                                            stride = [a//2 for a in stride], 
                                            padding = [(a-1)//2 for a in kernel_size], 
                                            norm_cfg= norm_cfg, 
                                            conv_type=conv_type, 
                                            bias = bias, 
                                            last_norm_on=False)

        # print('[PatchEmbed] project1', self.projection1) 
        # print('[PatchEmbed] project2', self.projection2) 
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_3tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w, pad_d = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w, input_d = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_d = input_d + pad_d
                input_size = (input_h, input_w, input_d)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            d_out = (input_size[2] + 2 * padding[2] - dilation[2] *
                     (kernel_size[2] - 1) - 1) // stride[2] + 1
            self.init_out_size = (h_out, w_out, d_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W, D). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w * out_d, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w, out_d).
        """
        
        if self.adap_padding:
            x = self.adap_padding(x)
        # print('[PatchEmbed] input', x.shape)
        x = self.projection1(x) 

        # print('[PatchEmbed] p1 ', x.shape)
        x = self.projection2(x)

        # print('[PatchEmbed] p2 ', x.shape)
        out_size = x.shape[2:]
        x = x.flatten(2).transpose(1, 2) # BCN > BNC
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class ConvLnGeluSequence(BaseModule):
    """
    conv-act-(flatten)-layernorm--conv
    ??
    """
    def __init__(self, in_channels, embed_dims, 
                    kernel_size = 3, 
                    stride = 2, padding = 1,
                    dilation = 1, bias = True, 
                    act_cfg = dict(type = 'GELU'),  
                    norm_cfg = dict(type = 'LN'),
                    conv_type = 'Conv3d', 
                    last_norm_on=True, 
                    init_cfg=None,):
        super(ConvLnGeluSequence, self).__init__(init_cfg = init_cfg)
        self.out_dim=embed_dims
        # self.conv1= nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride,padding=padding)

        self.conv1 = build_conv_layer(
                    dict(type=conv_type),
                    in_channels=in_channels,
                    out_channels=embed_dims,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=bias)

        # self.conv2=nn.Conv3d(embed_dims, embed_dims, kernel_size=3, stride=1,padding=1)
        self.conv2 = build_conv_layer(
                    dict(type=conv_type),
                    in_channels=embed_dims,
                    out_channels=embed_dims,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias)
        # self.activate=activate()
        self.activate = build_activation_layer(act_cfg)
        # self.norm1=norm(embed_dims)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.last_norm_on = last_norm_on  
        if last_norm_on:
            self.norm2=build_norm_layer(norm_cfg, embed_dims)[1]
            
    def forward(self,x):
        x=self.conv1(x)
        x=self.activate(x)
        #norm1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Ws, Wh, Ww)
        

        x=self.conv2(x)
        if self.last_norm_on:
            x=self.activate(x)
            #norm2
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Ws, Wh, Ww)
        return x
        
    


class PatchMergingNN(BaseModule):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=2,
                 padding=None,
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'), 
                 init_cfg=None):
        super().__init__(init_cfg = init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size

        kernel_size = to_3tuple(kernel_size)
        stride = to_3tuple(stride)
        dilation = to_3tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding3D(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of unfold
            padding = 0
        else:
            self.adap_padding = None
        padding = to_3tuple(padding)
        
        self.reduction = nn.Conv3d(in_channels, out_channels,
                                    kernel_size=kernel_size, 
                                    stride=stride)
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]

    def forward(self, x, input_size):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        Ht, Wt, Dt, = input_size
        B, L, C = x.shape
        assert L == Wt * Dt * Ht, "input feature has wrong size"
        
        x = F.gelu(x)
        x = self.norm(x)

        x = x.view(B, Ht, Wt, Dt, C)
        x = x.permute(0,4,1,2,3)

        if self.adap_padding:
            x = self.adap_padding(x)
            Ht, Wt, Dt = x.shape[2:]

        x = self.reduction(x)

        (Hr, Wr, Dr) = output_size = x.shape[2:]

        x = x.permute(0,2,3,4,1).view(B, Hr * Wr * Dr , -1)

        return x, output_size