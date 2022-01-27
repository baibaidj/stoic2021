'''
This code is borrowed from Serge-weihao/CCNet-Pure-Pytorch
'''

# Based on the implementation in this repo, I made the 3D version of criss-cross attention as follows. May this helps.
import torch
import torch.nn as nn
from torch.nn import Softmax
from mmcv.cnn import ConvModule, normal_init
from mmcv.runner import force_fp32
print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

def INF(B,H,W):
    return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
    # .cuda()

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim, verbose = True):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.verbose = verbose


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        # bchw > bwch, b*w-c-h > b*w-h-c
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        # bchw > bhcw, b*h-c-w > b*w-w-c
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        
        if self.verbose: print_tensor('q', proj_query)
        if self.verbose: print_tensor('qh', proj_query_H)
        if self.verbose: print_tensor('qw', proj_query_W)

        proj_key = self.key_conv(x)

        # bchw > bwch, b*w-c-h
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        # bchw > bhcw, b*h-c-w
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)

        if self.verbose: print_tensor('k', proj_key)
        if self.verbose: print_tensor('kh', proj_key_H)
        if self.verbose: print_tensor('kw', proj_key_W)

        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)

        # batch matrix-matrix
        inf_holder = self.INF(m_batchsize, height, width) # > bw-h-h
        if self.verbose: print_tensor('inf', inf_holder)
        energy_H = torch.bmm(proj_query_H, proj_key_H)+inf_holder
        energy_H = energy_H.view(m_batchsize,width,height,height).permute(0,2,1,3)

        if self.verbose: print_tensor('eh', energy_H)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        if self.verbose: print_tensor('ew', energy_W)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        if self.verbose: print_tensor('eall', concate)
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)

        if self.verbose: print_tensor('atth', att_H); print_tensor('attw', att_W)
        
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)

        if self.verbose: print_tensor('outh', out_H); print_tensor('outw', out_W)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x




def INF3D(B, H, W, D):
    return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W*D,1,1)

class CrissCrossAttention3D(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, query_channel, key_channel = None, out_channel = None,  embed_channel = None, 
                        is_add2query = True,
                        verbose = False):
        super(CrissCrossAttention3D, self).__init__()

        if embed_channel == None: embed_channel = query_channel //8
        if out_channel == None: out_channel = query_channel
        if key_channel == None: key_channel = query_channel
        self.is_add2query = is_add2query

        self.query_conv = nn.Conv3d(in_channels=query_channel, out_channels=embed_channel, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=key_channel, out_channels=embed_channel, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=key_channel, out_channels=out_channel, kernel_size=1)
        self.softmax = Softmax(dim=4)
        self.INF = INF3D
        self.gamma = nn.Parameter(torch.zeros(1))
        self.verbose = verbose

    @force_fp32(apply_to=('query',) , out_fp16=False)
    def forward(self, query, key = None):
        if key is None: key = query
        m_batchsize, _, height, width, depth= query.size()
        proj_query = self.query_conv(query)
        # bchw > bwch, b*w*d-c-h > b*w*d-h-c
        proj_query_H = proj_query.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize*width*depth,-1,height).permute(0, 2, 1)
        # bchw > bhcw, b*h*d-c-w > b*h*d-w-c
        proj_query_W = proj_query.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize*height*depth,-1,width).permute(0, 2, 1)
        # bchwd > bwch, b*h*w-c-d > b*h*w-d-c
        proj_query_D = proj_query.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize*height*width,-1,depth).permute(0, 2, 1)
        
        if self.verbose: print_tensor('q', proj_query)
        if self.verbose: print_tensor('qh', proj_query_H)
        if self.verbose: print_tensor('qw', proj_query_W)
        if self.verbose: print_tensor('qd', proj_query_D)

        proj_key = self.key_conv(key)

        # bchw > bwch, b*w*d-c-h
        proj_key_H = proj_key.permute(0,3,4,1,2).contiguous().view(m_batchsize*width*depth,-1,height)
        # bchw > bhcw, b*h*d-c-w
        proj_key_W = proj_key.permute(0,2,4,1,3).contiguous().view(m_batchsize*height*depth,-1,width)
        proj_key_D = proj_key.permute(0,2,3,1,4).contiguous().view(m_batchsize*height*width,-1,depth)

        if self.verbose: print_tensor('k', proj_key)
        if self.verbose: print_tensor('kh', proj_key_H)
        if self.verbose: print_tensor('kw', proj_key_W)
        if self.verbose: print_tensor('kd', proj_key_D)

        proj_value = self.value_conv(key)
        proj_value_H = proj_value.permute(0,3,4,1,2).contiguous().view(m_batchsize*width*depth,-1,height)
        proj_value_W = proj_value.permute(0,2,4,1,3).contiguous().view(m_batchsize*height*depth,-1,width)
        proj_value_D = proj_value.permute(0,2,3,1,4).contiguous().view(m_batchsize*height*width,-1,depth)

        # batch matrix-matrix
        inf_holder = self.INF(m_batchsize, height, width, depth).type_as(query).to(query.device) # > bw-h-h 
        if self.verbose: print_tensor('inf', inf_holder)
        energy_H = torch.bmm(proj_query_H, proj_key_H)+inf_holder # bwd-h-c, bwd-c-h > bwd-h-h
        energy_H = energy_H.view(m_batchsize,width,depth,height,height).permute(0,3,1,2,4) # bhwdh
        if self.verbose: print_tensor('eh', energy_H) 

        #  b*h*d-w-c, b*h*d-c-w > b*h*d-w-w
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, depth, width, width).permute(0, 1, 3, 2, 4) # 
        if self.verbose: print_tensor('ew', energy_W)
        
        energy_D = torch.bmm(proj_query_D, proj_key_D).view(m_batchsize, height, width, depth, depth)
        if self.verbose: print_tensor('ew', energy_W)


        concate = self.softmax(torch.cat([energy_H, energy_W, energy_D], 4)) # bhwd*(h+w+d)
        print_tensor('affinity matrix', concate)  #
        # bhw(H+W) > bhwH, bwhH; 
        att_H = concate[:,:,:,:,0:height].permute(0,2,3,1,4).contiguous().view(m_batchsize*width*depth,height,height)
        att_W = concate[:,:,:,:,height:height+width].permute(0,1,4,2,3).contiguous().view(m_batchsize*height*depth,width,width)
        att_D = concate[:,:,:,:,height+width:].contiguous().view(m_batchsize*height*width, depth, depth)

        if self.verbose: print_tensor('atth', att_H); print_tensor('attw', att_W);print_tensor('attd', att_D)

        # p-c-h, p-h-h > p-c-h
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,depth,-1,height).permute(0,3,4,1,2)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,depth,-1, width).permute(0,3,1,4,2)
        out_D = torch.bmm(proj_value_D, att_D.permute(0, 2, 1)).view(m_batchsize,height, width, -1, depth).permute(0,3,1,2,4)

        if self.verbose: print_tensor('outh', out_H); print_tensor('outw', out_W), print_tensor('outd', out_D)
        #print(out_H.size(),out_W.size())
        att_result = self.gamma*(out_H + out_W + out_D)
        if self.is_add2query: return att_result + query
        else: return att_result 




def criss_cross_operation_3d(proj_query, proj_key, proj_value, verbose = False):
    assert proj_query.dim() == 5 and proj_key.dim() == 5 and proj_value.dim() == 5
    m_batchsize, _, height, width, depth= proj_query.size()
    tensor_dim = proj_query.dim()
    inf_tensor = INF3D
    
    # bchw > bwch, b*w*d-c-h > b*w*d-h-c
    proj_query_H = proj_query.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize*width*depth,-1,height).permute(0, 2, 1)
    # bchw > bhcw, b*h*d-c-w > b*h*d-w-c
    proj_query_W = proj_query.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize*height*depth,-1,width).permute(0, 2, 1)
    # bchwd > bwch, b*h*w-c-d > b*h*w-d-c
    proj_query_D = proj_query.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize*height*width,-1,depth).permute(0, 2, 1)
    if verbose: print_tensor('q', proj_query)
    if verbose: print_tensor('qh', proj_query_H)
    if verbose: print_tensor('qw', proj_query_W)
    if verbose: print_tensor('qd', proj_query_D)

    # bchw > bwch, b*w*d-c-h
    proj_key_H = proj_key.permute(0,3,4,1,2).contiguous().view(m_batchsize*width*depth,-1,height)
    # bchw > bhcw, b*h*d-c-w
    proj_key_W = proj_key.permute(0,2,4,1,3).contiguous().view(m_batchsize*height*depth,-1,width)
    proj_key_D = proj_key.permute(0,2,3,1,4).contiguous().view(m_batchsize*height*width,-1,depth)
    if verbose: print_tensor('k', proj_key)
    if verbose: print_tensor('kh', proj_key_H)
    if verbose: print_tensor('kw', proj_key_W)
    if verbose: print_tensor('kd', proj_key_D)

    proj_value_H = proj_value.permute(0,3,4,1,2).contiguous().view(m_batchsize*width*depth,-1,height)
    proj_value_W = proj_value.permute(0,2,4,1,3).contiguous().view(m_batchsize*height*depth,-1,width)
    proj_value_D = proj_value.permute(0,2,3,1,4).contiguous().view(m_batchsize*height*width,-1,depth)

    # batch matrix-matrix
    inf_holder = inf_tensor(m_batchsize, height, width, depth).type_as(proj_query).to(proj_query.device) # > bw-h-h 
    if verbose: print_tensor('inf', inf_holder)
    energy_H = torch.bmm(proj_query_H, proj_key_H)+inf_holder # bwd-h-c, bwd-c-h > bwd-h-h
    energy_H = energy_H.view(m_batchsize,width,depth,height,height).permute(0,3,1,2,4) # bhwdh
    if verbose: print_tensor('eh', energy_H) 

    #  b*h*d-w-c, b*h*d-c-w > b*h*d-w-w
    energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, depth, width, width).permute(0, 1, 3, 2, 4) # 
    if verbose: print_tensor('ew', energy_W)
    
    energy_D = torch.bmm(proj_query_D, proj_key_D).view(m_batchsize, height, width, depth, depth)
    if verbose: print_tensor('ed', energy_D)

    concate = torch.softmax(torch.cat([energy_H, energy_W, energy_D], tensor_dim - 1), tensor_dim - 1) # bhwd*(h+w+d)
    if verbose: print_tensor('eall', concate) 
    # print_tensor('CC3D affinity matrix', concate)
    # bhw(H+W) > bhwH, bwhH; 
    att_H = concate[:,:,:,:,0:height].permute(0,2,3,1,4).contiguous().view(m_batchsize*width*depth,height,height)
    att_W = concate[:,:,:,:,height:height+width].permute(0,1,4,2,3).contiguous().view(m_batchsize*height*depth,width,width)
    att_D = concate[:,:,:,:,height+width:].contiguous().view(m_batchsize*height*width, depth, depth)

    if verbose: print_tensor('atth', att_H); print_tensor('attw', att_W);print_tensor('attd', att_D)

    # p-c-h, p-h-h > p-c-h
    out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,depth,-1,height).permute(0,3,4,1,2)
    out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,depth,-1, width).permute(0,3,1,4,2)
    out_D = torch.bmm(proj_value_D, att_D.permute(0, 2, 1)).view(m_batchsize,height, width, -1, depth).permute(0,3,1,2,4)

    if verbose: print_tensor('outh', out_H); print_tensor('outw', out_W), print_tensor('outd', out_D)
    
    return out_H + out_W + out_D




def criss_cross_operation_2d(proj_query, proj_key, proj_value,  verbose = False):
    assert proj_query.dim() == 4 and proj_key.dim() == 4 and proj_value.dim() == 4
    tensor_dim = proj_query.dim()
    m_batchsize, _, height, width, depth= proj_query.size()
    inf_tensor = INF

    # bchw > bwch, b*w-c-h > b*w-h-c
    proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
    # bchw > bhcw, b*h-c-w > b*w-w-c
    proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)

    
    if verbose: print_tensor('q', proj_query)
    if verbose: print_tensor('qh', proj_query_H)
    if verbose: print_tensor('qw', proj_query_W)

    # bchw > bwch, b*w-c-h
    proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
    # bchw > bhcw, b*h-c-w
    proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)

    if verbose: print_tensor('k', proj_key)
    if verbose: print_tensor('kh', proj_key_H)
    if verbose: print_tensor('kw', proj_key_W)

    proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
    proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)

    # batch matrix-matrix
    inf_holder = inf_tensor(m_batchsize, height, width).type_as(proj_query).to(proj_query.device) # > bw-h-h 
    if verbose: print_tensor('inf', inf_holder)
    energy_H = torch.bmm(proj_query_H, proj_key_H)+inf_holder # bwd-h-c, bwd-c-h > bwd-h-h
    energy_H = energy_H.view(m_batchsize,width,height,height).permute(0,2,1,3)

    if verbose: print_tensor('eh', energy_H) 

    #  b*h*d-w-c, b*h*d-c-w > b*h*d-w-w
    energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
    if verbose: print_tensor('ew', energy_W)

    concate = torch.softmax(torch.cat([energy_H, energy_W, ], tensor_dim - 1), tensor_dim - 1) # bhwd*(h+w+d)
    if verbose: print_tensor('eall', concate) 
    # bhw(H+W) > bhwH, bwhH; 
    att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
    att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)

    if verbose: print_tensor('atth', att_H); print_tensor('attw', att_W)
    
    out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
    out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)

    if verbose: print_tensor('outh', out_H); print_tensor('outw', out_W), print_tensor('outd', out_D)
    
    return out_H + out_W

class _CCNonLocalNd(nn.Module):
    """Basic Non-local module.

    This module is proposed in
    "Non-local Neural Networks"
    Paper reference: https://arxiv.org/abs/1711.07971
    Code reference: https://github.com/AlexHex7/Non-local_pytorch

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            `1/sqrt(inter_channels)` when the mode is `embedded_gaussian`.
            Default: True.
        conv_cfg (None | dict): The config dict for convolution layers.
            If not specified, it will use `nn.Conv2d` for convolution layers.
            Default: None.
        norm_cfg (None | dict): The config dict for normalization layers.
            Default: None. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: embedded_gaussian.
    """

    def __init__(self,
                 in_channels,
                 out_channels = None, 
                 key_channels = None, 
                 reduction=8,
                 use_scale=True,
                 conv_cfg=dict(type = 'Conv3d'),
                 norm_cfg=dict(type = 'BN3d', requires_grad=True),
                 use_outconv = False,
                 mode='embedded_gaussian',
                 verbose = False,
                 **kwargs):
        super(_CCNonLocalNd, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.out_channels = in_channels if out_channels is None else out_channels
        self.key_channels = in_channels if key_channels is None else key_channels
        self.mode = mode
        self.use_outconv = use_outconv
        self.gamma = nn.Parameter(torch.zeros(1))

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.verbose = verbose

        if mode not in [
                'gaussian', 'embedded_gaussian', 'dot_product', 'concatenation'
        ]:
            raise ValueError("Mode should be in 'gaussian', 'concatenation', "
                             f"'embedded_gaussian' or 'dot_product', but got "
                             f'{mode} instead.')

        self.is_3d = False if self.conv_cfg is None else (True if '3d' in self.conv_cfg.get('type', '').lower() else False)
        # g, theta, phi are defaulted as `nn.ConvNd`. 
        # Here we use ConvModule for potential usage. theta4query phi4key g4value 
        self.g = ConvModule(
            self.key_channels,
            self.out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)

        if self.use_outconv: 
            self.conv_out = ConvModule(
                self.inter_channels,
                self.in_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)

        # if self.mode != 'gaussian':
        self.theta = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)
        self.phi = ConvModule(
            self.key_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)

        # if self.mode == 'concatenation':
        #     self.concat_project = ConvModule(
        #         self.inter_channels * 2,
        #         1,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         bias=False,
        #         act_cfg=dict(type='ReLU'))

        self.init_weights(**kwargs)

    def init_weights(self, std=0.01, zeros_init=True):
        # if self.mode != 'gaussian':
        for m in [self.g, self.theta, self.phi]:
            normal_init(m.conv, std=std)
        # else:
        #     normal_init(self.g.conv, std=std)
        # if zeros_init:
        #     if self.conv_out.norm_cfg is None:
        #         constant_init(self.conv_out.conv, 0)
        #     else:
        #         constant_init(self.conv_out.norm, 0)
        # else:
        #     if self.conv_out.norm_cfg is None:
        #         normal_init(self.conv_out.conv, std=std)
        #     else:
        #         normal_init(self.conv_out.norm, std=std)


    def gaussian(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def embedded_gaussian(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def concatenation(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.size()
        pairwise_weight = pairwise_weight.view(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]

        return pairwise_weight

    # @force_fp32(apply_to=('query',) , out_fp16=False)
    def forward(self, query, key = None):
        # Assume `reduction = 1`, then `inter_channels = C`
        # or `inter_channels = C` when `mode="gaussian"`
        ip_dtype = query.dtype
        cc_func = criss_cross_operation_3d if self.is_3d else criss_cross_operation_2d
        if key is None: key = query
        # NonLocal1d x: [N, C, H]
        # NonLocal2d x: [N, C, H, W]
        # NonLocal3d x: [N, C, T, H, W]
        # NonLocal1d g_x: [N, H, C]
        # NonLocal2d g_x: [N, HxW, C]
        # NonLocal3d g_x: [N, TxHxW, C]
        # g_x = self.g(query).view(n, self.inter_channels, -1)
        # g_x = g_x.permute(0, 2, 1)
        proj_query = self.theta(query)        
        proj_key = self.phi(key)
        proj_value = self.g(key)

        cc_out = cc_func(proj_query.float(), proj_key.float(), proj_value.float(), self.verbose)
        cc_out = cc_out.to(ip_dtype)
        #print(out_H.size(),out_W.size())
        att_result = self.gamma*(cc_out)

        output = query + att_result #self.conv_out(att_result)

        return output


class CCAttention3D(nn.Module):

    """Basic Non-local module.

    This module is proposed in
    "Non-local Neural Networks"
    Paper reference: https://arxiv.org/abs/1711.07971
    Code reference: https://github.com/AlexHex7/Non-local_pytorch

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            `1/sqrt(inter_channels)` when the mode is `embedded_gaussian`.
            Default: True.
        conv_cfg (None | dict): The config dict for convolution layers.
            If not specified, it will use `nn.Conv2d` for convolution layers.
            Default: None.
        norm_cfg (None | dict): The config dict for normalization layers.
            Default: None. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: embedded_gaussian.
    """

    def __init__(self,
                  in_channels,
                 *args, 
                 repeat_times = 3, 
                 conv_cfg = dict(type = 'Conv3d'), 
                 norm_cfg=dict(type = 'BN3d', requires_grad=True),
                 act_cfg = dict(type='ReLU', inplace = True),
                 out_projection = True,
                 **kwargs):
        super(CCAttention3D, self).__init__()
        self.repeat_times = repeat_times
        self.att_blocks = nn.ModuleList([_CCNonLocalNd(in_channels, *args, conv_cfg=conv_cfg, norm_cfg=norm_cfg, 
                                                        **kwargs) for _ in range(repeat_times)])
        if out_projection: 
            self.out_projection = self.build_project(in_channels, in_channels, conv_cfg, norm_cfg, act_cfg)
        else:
            self.out_projection = None

    def build_project(self, in_channels, channels,
                      conv_cfg, norm_cfg, act_cfg, num_convs = 1):
        """Build projection layer for key/query/value/out."""
        convs = [
            ConvModule(
                in_channels,
                channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        ]
        for _ in range(num_convs - 1):
            convs.append(
                ConvModule(
                    channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query, key = None):
        # ip_dtype = query.dtype
        output = query
        for i in range(self.repeat_times):
            # print(f'[CCA3D] round {i}')
            output = self.att_blocks[i](output, key)

        if self.out_projection:
            output = self.out_projection(output)
        return output

if __name__ == '__main__':
    # model = CrissCrossAttention(16)
    # x = torch.randn(2, 16, 5, 6)
    # out = model(x)
    # print(out.shape)

    # model = CrissCrossAttention3D(16)
    model = CCAttention3D(16, repeat_times=3)
    x = torch.randn(2, 16, 5, 6, 7)
    out = model(x)
    print(out.shape)