_base_ = [
    '../datasets/stoic2021_lung_256x224x224.py',
]
# model settings
conv_cfg = dict(type = 'Conv3d')
norm_cfg = dict(type='IN3d', requires_grad=True) 
stem_channels = 16
model = dict(
    type='ImageClassifierMed',
    backbone=dict(
        type='ResNet3dIso', # verbose = False, 
        deep_stem = True,
        avg_down=True,
        depth='343d', # 18.3G 
        in_channels=1,
        stem_stride_1 = 2,
        stem_stride_2 = 1, 
        stem_channels= stem_channels, # 16 
        base_channels= stem_channels * 2, # 32 
        num_stages=4,
        strides=(2, 2, 2, 2), # 32, 64, 128, 256
        dilations=(1, 1, 1, 1),
        out_indices=(1, 2, 3, 4, 5), # 2, 4, 8, 16, 32
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        style='pytorch',
        ),
    head=dict(type='LinearClsHead',
            in_channels = stem_channels * 16,
            num_classes = 2,
            add_feat_dist = False, 
            age_encoding= dict(),
            # is_multi_task = False, 
            loss=dict(type='CrossEntropyLoss', 
                        use_sigmoid = True, 
                        loss_weight=2.0, 
                        # class_weight = (0.8, 1.0)
                        ),
            in_index = -1, 
            dim = 3, dropout_ratio = 0.1), 

    gpu_aug_pipelines = {{ _base_.gpu_aug_pipelines }},
    train_cfg = dict(), 
    test_cfg = dict(),
)


    # detections_per_img = plan_arch.get("detections_per_img", 100)
    # score_thresh = plan_arch.get("score_thresh", 0)
    # topk_candidates = plan_arch.get("topk_candidates", 10000)
    # remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
    # nms_thresh = plan_arch.get("nms_thresh", 0.6)