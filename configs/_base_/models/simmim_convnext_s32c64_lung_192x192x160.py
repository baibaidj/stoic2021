_base_ = [
    '../datasets/allct_lung_160x160x128.py',
]
num_classes = 1 # this is an RPN 
# model settings
conv_cfg = dict(type = 'Conv3d')
norm4head = dict(type='GN', num_groups=16, requires_grad=True) 
norm_cfg = dict(type='IN3d', requires_grad=True) 
# bs=2, ng=8, chn=2, m=18.1G;  bs=2, ng=2, m=17.2G;  bs=2, ng=8, chn=1, m=19.3G 
stem_channels = 64
stem_stride = 2
fpn_channel = 128 # > 128 #stem_channels * (2**3)
model = dict(
    type='SimMIM',
    backbone=dict(
        type='ConvNeXt3D4SimMIM',
        in_channels=1, 
        stem_cfg = dict(conv1kernel = 5, conv1stride = 2, conv1_chn_div = 2, 
                        conv2kernel = 5, conv2stride = 1), 
        expand_ratio = 4, 
        dw_kernel_size = 7, 
        num_stages=5,
        depths=[0, 4, 4, 8, 3], 
        dims=[64, 64, 128, 256, 320],  # 2, 4, 8, 16, 32
        drop_path_rate=0.2, 
        layer_scale_init_value=1.0, 
        out_indices=(0, 1, 2, 3, 4),
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,  # TODO: replace ReLU with Swish
        ),
    neck=dict(
        type='FPN3D2022',
        in_channels=[64, 64, 128, 256, 320], 
        fixed_out_channels = fpn_channel,
        start_level=2,
        conv_cfg = conv_cfg, 
        norm_cfg = norm_cfg, 
        kernel_size = 5, 
        add_extra_convs=False,
        is_double_chn = False, 
        num_outs=5, 
        upsample_cfg=dict(type=None, mode='trilinear', use_norm = True, 
                    kernel_size = (2,2,2), stride = (2,2,2)),
        ),

    seg_head = dict(
        type='FCNHead3D', # verbose = True, 
        in_channels= 64,
        in_index= 1,
        channels= stem_channels//2,
        # input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes= 1,
        conv_cfg = conv_cfg, 
        norm_cfg=norm4head,
        align_corners=False,
        gt_index = 0),
    gpu_aug_pipelines = {{ _base_.gpu_aug_pipelines }},
    mask_cfg = dict(input_size={{ _base_.patch_size }}, mask_patch_size=32,
                                 stem_stride=stem_stride, mask_ratio=0.5), 
                                 
    test_cfg=dict(roi_size = {{ _base_.patch_size }}, sw_batch_size = 6,
        blend_mode = 'constant' , overlap=0.01, sigma_scale = 0.125, # 'gaussian or constant
        padding_mode='constant' )
)


    # detections_per_img = plan_arch.get("detections_per_img", 100)
    # score_thresh = plan_arch.get("score_thresh", 0)
    # topk_candidates = plan_arch.get("topk_candidates", 10000)
    # remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
    # nms_thresh = plan_arch.get("nms_thresh", 0.6)