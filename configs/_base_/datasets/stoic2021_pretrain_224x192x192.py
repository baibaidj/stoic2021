# dataset settings
dataset_type = 'AllCTDataset'
img_dir = 'data/STOIC2021Round1'
in_channel=  1
norm_param = {
    "mean": -86.8, "std": 443.8, "median": 111,
    "mn": -1133, "mx": 2185,
    "percentile_99_5": 505,
    "percentile_00_5": -1024}

keys = ('img', ) #, seg='instance_seg' 
dtypes = ('float', ) # , 'float', 
interp_modes = ("bilinear", )  #  , "bilinear", 'nearest'
core_key_num = 1
ext_patch_size = (256, 224, 224) # avoid artifacts such as boarder reflection
patch_size = (224, 192, 192)  # [160 192 112] # xyz
train_pipeline = [
    dict(type = 'LoadImaged', keys = keys, reader = 'NibabelReader'),  # img_meta_dict see mmseg.datasets.pipeline.transform_moani
    dict(type = 'AddChanneld', keys= keys), 
    # dict(type = 'ConvertLabeld', keys = 'gt_semantic_seg',  label_mapping = label_mapping), 
    dict(type = 'RandCropByLabelBBoxRegiond', keys=keys,
                                        label_key='img',
                                        spatial_size= ext_patch_size, 
                                        num_samples=1,
                                        pos=2, neg=1,
                                        whole_image_as_bg=1,
                                        percentile_00_5=-900,
                                        return_keys= 'all', #['image', 'label'], 
                                        # data_info_csv = f'{img_dir}/case_info_records_skeleton.csv',
                                        custom_center_kargs = {'base_value': None, 'jitter': (8, 8, 8)}
                                        ),
    dict(type = 'SpatialPadd_', keys=keys, spatial_size= ext_patch_size, # padshape
                                mode='reflect', verbose = False),  
    dict(type = 'CastToTyped_', keys = keys,  dtype=dtypes), 
    dict(type = 'ToTensord', keys = keys),
    dict(type = 'RandFlipd_', keys = keys, spatial_axis=(0, 1), prob=0.25),
    dict(type = 'RandFlipd_', keys = keys, spatial_axis=(2, ), prob=0.25),
    # dict(type = 'RideOnLabel', keys = {'seg': ('seg', 'skeleton') }, cat_dim = 0),
    # dict(type = 'DataStatsd', keys = keys, prefix = 'Final'), 
    dict(type='FormatShapeMonai', verbose = False, keys = keys[:core_key_num],  channels = in_channel),
    dict(type='Collect', keys=keys[:core_key_num], verbose = False, meta_keys = ('img_meta_dict', )),
]
test_keys = ('img', )
test_pipeline = [
        dict(type = 'LoadImaged', keys = keys, reader = 'NibabelReader'),
        dict(type='MultiScaleFlipAug3D',
            # label_mapping = label_mapping,
            # value4outlier = 1,
            target_spacings = None, 
            flip=False,
            flip_direction= ['diagonal'],
            transforms=[
                dict(type = 'AddChanneld', keys= test_keys), 
                dict(type = 'SpacingTTAd', keys = test_keys, pixdim = None, mode = ('bilinear',)),
                dict(type = 'SpatialPadd_', keys= test_keys, spatial_size= patch_size, mode='reflect', method = 'end'), 
                dict(type = 'FlipTTAd_', keys = test_keys),  
                dict(type = 'CastToTyped_', keys = 'img',  dtype= ('float', )),
                dict(type = 'ToTensord', keys = 'img'), 
                dict(type = 'NormalizeIntensityGPUd', keys='img',
                            subtrahend=norm_param['mean'],
                            divisor=norm_param['std'],
                            percentile_99_5=norm_param['percentile_99_5'],
                            percentile_00_5=norm_param['percentile_00_5']
                            ),
                dict(type= 'FormatShapeMonai', keys=['img'], use_datacontainer = False,
                                             channels = in_channel),
                dict(type='Collect', keys=['img'], meta_keys = ('img_meta_dict', ))
                ], 
            )
]

# draw_step = 8
total_samples = 160 #// draw_step #9842
sample_per_gpu = 3# bs2 >> 24.5 G  # 
train_sample_rate = 1.0
val_sample_rate = 0.30

data = dict(
    samples_per_gpu=sample_per_gpu,  # 16-3G
    workers_per_gpu= 6, 
    persistent_workers = True,
    train=dict(
        type=dataset_type, img_dir=img_dir, 
        sample_rate = train_sample_rate, split='train',
        pipeline=train_pipeline,  
        fn2imglist = 'case_info_szall.csv',
        ),
    val=dict(
        type=dataset_type, img_dir=img_dir, 
        sample_rate = val_sample_rate, split='test', 
        pipeline=test_pipeline,
        fn2imglist = 'case_info_szall.csv',
        # fn_spliter = ['-', 0]
        ),
    test=dict(
        type=dataset_type, img_dir=img_dir, 
        sample_rate = 1.0, split='test', 
        pipeline=test_pipeline, 
        fn2imglist = 'case_info_szall.csv',
        ))

gpu_aug_pipelines = [
        # # data in torch tensor, not necessarily on gpu
        dict(type = 'Rand3DElasticGPUd', keys=keys[:core_key_num],
            sigma_range=(9, 13), # larger sigma mean smoother offset with smaller values
            magnitude_range=(64, 256), # s=8, (-0.008, 0.006) * 256 > (2.04, 1.53)
            spatial_size=patch_size, 
            rotate_range=[15] * 3, #rotate_angle * np.pi / 180.0
            translate_range = [4] * 3, 
            scale_range=[0.2] * 3,
            prob=1.0,
            mode=interp_modes[:core_key_num], 
            verbose=False
            ),  # 2s/mini-batch
        dict(type = 'NormalizeIntensityGPUd', keys='img',
                    subtrahend=norm_param['mean'],
                    divisor=norm_param['std'],
                    percentile_99_5=norm_param['percentile_99_5'],
                    percentile_00_5=norm_param['percentile_00_5']
                    ),
        dict(type = 'RandGaussianNoised_', keys='img', prob=0.4, std=0.05), 
        # dict(type = 'SaveImaged', keys = keys[:core_key_num], 
        #     output_dir = f'work_dirs/simmim_convnext_s32c64em4_lung_224x224x192_100eps/debug', 
        #     resample = False, save_batch = True, on_gpu = True),    
        ]