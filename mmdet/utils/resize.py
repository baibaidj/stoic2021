import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def resize_3d(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            # input_h, input_w = input.shape[2:]
            in_shape_list = input.shape[-3:] if len(size)==3 else input.shape[-2:]            
            # output_h, output_w = size
            out_shape_list = size
            if any([out_shape_list[i] > in_shape_list[i] for i in range(len(in_shape_list))]):
                if ((all([a > 1 for a in tuple(in_shape_list+ out_shape_list)])) and 
                    all([(out_shape_list[i] - 1) % (in_shape_list[i] - 1) for i in range(len(in_shape_list))])
                    ):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {tuple(in_shape_list)} is `x+1` and '
                        f'out size {tuple(out_shape_list)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)
            
class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            if isinstance(self.scale_factor, tuple):
                size = [int(t * self.scale_factor[i]) for i, t in enumerate(x.shape[2:])]
            else:
                size = [int(t * self.scale_factor) for i, t in enumerate(x.shape[2:])]

        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)



def bnchw2bchw(imgs, use_tsm = True):
    """
    reshape 5d imgs to 4d, where the 1st (batchsize) and 2nd (temporal dimension) will be collapsed
    imgs: B, N, C, H, W >> B*N, C, H, W

    not friendly for metric calculation of seg mask, which assumes the class/channel dimension is on the second
    i.e. B,C,T,H,W. reshape from BNCHW to BCNHW is not trivial. 
    """
    if use_tsm and imgs.dim() == 5:
        batches = imgs.shape[0]
        # B, N, C, H, W >> B*N, C, H, W
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches # N is num_segments
    else:
        num_segs = None
    return imgs, num_segs

print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

def list_dict2dict_list(list_dict, verbose = False):
    dict_list = {key: list() for key in list_dict[0]}
    tensor_keys = set()
    for i, data in enumerate(list_dict):
        for key, val in data.items():
            if isinstance(val, list):
                dict_list[key].extend(val)
            else:
                dict_list[key].append(val)

            if isinstance(val, torch.Tensor): 
                if verbose: print_tensor(f'[list2dict] {i} {key}', val)
                tensor_keys.add(key)
    # concat tensors into mini-batch
    for k in tensor_keys:
        if verbose: print(f'[Concat] {k} len', len(dict_list[k]))
        dict_list[k] = torch.cat(dict_list[k], axis = 0)    
    return dict_list



# class Reshape4BNCHW():
#     def __init__(self, reverse = False, is_apply = False, temporal_size = 8):
#         self.reverse = reverse
#         self.is_apply = is_apply
#         self.temporal_size = temporal_size

    
#     def __call__(self, imgs):
    
#         if self.reverse: 
            
#             bn = imgs.shape[0]
#             imgs = imgs.view((bn//self.temporal_size, self.temporal_size) + imgs.shape[] )
            

#         if self.is_apply and imgs.dim() == 5:
#             batches = imgs.shape[0]
#             # B, N, C, H, W >> B*N, C, H, W
#             imgs = imgs.reshape((-1, ) + imgs.shape[2:])
#             num_segs = imgs.shape[0] // batches # N is num_segments
#         else:
#             num_segs = None
    
        