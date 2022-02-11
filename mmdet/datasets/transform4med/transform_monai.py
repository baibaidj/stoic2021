
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    CastToTyped_,
    SpatialPadd_,
    ToTensord,

    # self definition
    NormalizeIntensityGPUd,
    RandCropByLabelBBoxRegiond,
    # RandCropByPosNegMultiLabeld,
    RandFlipd_,
    RandGaussianNoised_,

    DataStatsd,
    CenterSpatialCropDJ,
    SaveImaged, 
    Rand3DElasticGPUd, 
    SpacingTTAd, 
    FlipTTAd_
)

from ..builder import PIPELINES 

preg = PIPELINES.register_module()
LoadImaged = preg(LoadImaged)
AddChanneld = preg(AddChanneld)
CastToTyped_ = preg(CastToTyped_)
SpatialPadd_ = preg(SpatialPadd_)
ToTensord = preg(ToTensord)

NormalizeIntensityGPUd = preg(NormalizeIntensityGPUd)
RandCropByLabelBBoxRegiond = preg(RandCropByLabelBBoxRegiond)
CenterSpatialCropDJ = preg(CenterSpatialCropDJ)
Rand3DElasticGPUd = preg(Rand3DElasticGPUd)

RandGaussianNoised_ = preg(RandGaussianNoised_)
RandFlipd_ = preg(RandFlipd_)

DataStatsd = preg(DataStatsd)
SpacingTTAd = preg(SpacingTTAd)
FlipTTAd_ = preg(FlipTTAd_)
SaveImaged = preg(SaveImaged)


@PIPELINES.register_module()
class AddInfo2Meta():

    AGE_MAP = {35: 1, 45: 2, 55: 3, 65: 4, 75: 5, 85: 6}
    SEX_MAP = {'F': 0, 'M': 1, 'A': 2, 'O': 2, 'N': 2}

    def __init__(self, key = 'img_meta_dict', sub_key = 'filename_or_obj') -> None:
        self.key = key
        self.sub_key = sub_key
        pass

    def __call__(self, data):
        d = dict(data)
        img_fp = d[self.key][self.sub_key]
        # try:
        target_cls, age_num, sex_num, pid = self.stoic_class_from_fname(img_fp)
        # except KeyError:
        #     print('[LoadError] caseid', img_fp)
        d[self.key]['target_class'] = target_cls
        d[self.key]['age'] = age_num
        d[self.key]['sex'] = sex_num
        return d

    def stoic_class_from_fname(self, abs_path):
        """
        covid = case_info['probCOVID']
        severity = case_info['probSevere']
        store_file = f'{pid}_age{age}_sex{sex}_covid{covid}_severe{severity}'
        
        """
        fname = abs_path.split('/')[-1].split('.')[0]
        pid, age, sex, covid, severe = fname.split('_')
        age = age[3:]
        sex = sex[3:]
        covid = int(covid[5:])
        severe = int(severe[6:])
        age_num = self.AGE_MAP[int(age[1:-1]) if len(age) > 2 else int(age)]
        sex_num = self.SEX_MAP.get(sex, 2)
        target_cls = [covid, severe]
        return target_cls, age_num, sex_num, pid

# dict_keys(['image', 'label', 'dtmap', 'skeleton', 'seg_fields', 'image_meta_dict', 'label_meta_dict', 'dtmap_meta_dict', 'skeleton_meta_dict'])
# image_meta_dict example: 
# {'sizeof_hdr': array(348, dtype=int32), 
# 'extents': array(0, dtype=int32), 
# 'session_error': array(0, dtype=int16), 
# 'dim_info': array(0, dtype=uint8), 
# 'dim': array([  3, 441, 441, 234,   1,   1,   1,   1], dtype=int16), 
# 'in$ent_p1': array(0., dtype=float32), 
# 'intent_p2': array(0., dtype=float32), 
# 'intent_p3': array(0., dtype=float32), 
# 'intent_code': array(0, dtype=int16),
# 'datatype': array(4, dtype=int16), 
# 'bitpix': array(16, dtype=int16), 
# '$lice_start': array(0, dtype=int16), 
# 'pixdim': array([1.   , 0.725, 0.725, 1.   , 0.   , 0.   , 0.   , 0.   ], dtype=float32), 
# 'vox_offset': array(0., dtype=float32), 
# 'scl_slope': array(nan, dtype=float32), 
# 'scl_inter': array(nan, dtype=float32), 
# 'slice_end': array(0, dtype=int16), 
# 'slice_code': array(0, dtype=uint8), 
# 'xyzt_$nits': array(2, dtype=uint8), 
# 'cal_max': array(0., dtype=float32), 
# 'cal_min': array(0., dtype=float32), 
# 'slice_duration': array(0., dtype=float32), 
# 'toffset': array(0., dtype=float32), 'glmax': array(0, dtype=int32), 
# 'glm$n': array(0, dtype=int32), 'qform_code': array(1, dtype=int16), 
# 'sform_code': array(0, dtype=int16), 
# 'quatern_b': array(0., dtype=float32), 
# 'quatern_c': array(0., dtype=float32), 
# 'quatern_d': array(1., dtype=float32), 
# 'qo$fset_x': array(-0., dtype=float32), 
# 'qoffset_y': array(-0., dtype=float32), 
# 'qoffset_z': array(0., dtype=float32), 
# 'srow_x': array([0., 0., 0., 0.], dtype=float32), 
# 'srow_y': array([0., 0., 0., 0.], dtype=float32), 
# 'srow_z': array([0., 0., 0., 0.], dtype=float32), 
# 'affine': array(
#       [[-0.72500002,  0.        ,  0.        , -0.        ],
#        [ 0.        , -0.72500002,  0.        , -0.        ],
#        [ 0.        ,  0.        ,  1.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]]), 
# 
# 'original_affine': array(
#       [[-0.72500002,  0.        ,  0.        , -0.        ],
#        [ 0.        , -0.72500002,  0.        , -0.        ],
#        [ 0.        ,  0.        ,  1.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]]), 
# 'as_closest_canonical': False, 
# 'spatial_shape': array([441, 441, 234], dtype=int16), 
# 'filename_or_obj': 'data/Task011_abdomen_anatomy/imagesTr/case_00489_0000.nii.gz'}