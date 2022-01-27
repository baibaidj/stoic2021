_base_ = [
    '../datasets/stoic2021_lung_224x224x192.py',
]
num_classes_cls = 2
# model settings
conv_cfg = dict(type = 'Conv3d')
norm4head = dict(type='GN', num_groups=16, requires_grad=True) 
norm_cfg = dict(type='IN3d', requires_grad=True) 
model = dict(
    type='ImageClassifierMed',
    backbone=dict(
        type='ConvNeXt3D',
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
    # neck= dict(type = 'GlobalAveragePooling', dim = 3) , 
    head=dict(type='LinearClsHead',
            in_channels = 320,
            num_classes = 2,
            loss=dict(type='FocalLossMultitask', 
                        use_sigmoid = True, 
                        loss_weight=2.0, 
                        class_weight = (1.0, 1.0)),
            in_index = -1, 
            dim = 3, dropout_ratio = 0.15), 

    gpu_aug_pipelines = {{ _base_.gpu_aug_pipelines }},
    test_cfg = None,
    train_cfg = None, 
)


    # detections_per_img = plan_arch.get("detections_per_img", 100)
    # score_thresh = plan_arch.get("score_thresh", 0)
    # topk_candidates = plan_arch.get("topk_candidates", 10000)
    # remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
    # nms_thresh = plan_arch.get("nms_thresh", 0.6)