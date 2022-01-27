import os
import os.path as osp
import numpy as np
import nibabel as nb
from PIL import Image
from pathlib import Path
import time, json, random
import SimpleITK as sitk
import pdb, codecs, re

print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

def save2json(obj, data_rt, filename, indent=4, sort_keys=True):
    assert 'json' in filename
    fp = osp.join(data_rt, filename)
    with open(fp, 'w', encoding='utf8') as f:
        json.dump(obj, f, sort_keys=sort_keys, ensure_ascii=False, indent=indent)
    return fp

def load2json(json_fp):
    assert osp.exists(json_fp)
    data = dict()
    if os.path.exists(json_fp):
        with open(json_fp, 'r') as f:
            data = json.load(f)
    return data


def save_string_list(file_path, l, is_utf8=False):
    """
    Save string list as mitok file
    - file_path: file path
    - l: list store strings
    """
    file_path = str(file_path)
    l = [l] if type(l) is not list else l
    if is_utf8:
        f = codecs.open(file_path, 'w', 'utf-8')
    else:
        f = open(file_path, 'w')
    nb_l = len(l)
    for i, item in enumerate(l):
        item = str(item)
        line = item if i == nb_l- 1 else (item + '\n')
        f.write(line)
    f.close()


def remove_white_space(strings):
    try:
        result = re.sub(r'[\(\)\t\n\s]', '', strings)
    except:
        result = strings
    return result


def load_string_list(file_path, is_utf8=False):
    """
    Load string list from mitok file
    """
    try:
        if is_utf8:
            f = codecs.open(file_path, 'r', 'utf-8')
        else:
            f = open(file_path)
        l = []
        for item in f:
            item = item.strip()
            if len(item) == 0:
                continue
            l.append(item)
        f.close()
    except IOError:
        print('open error %s' % file_path)
        return None
    else:
        return l

def convert_label(label, label_mapping = None, inverse=False, value4outlier = 0):
    if label_mapping is None:
        return label
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    
    if not inverse:
        max_value = max([v for _, v in label_mapping.items()])
        label[label > max_value] = value4outlier
    return label

def open_mask_random(fp):
    """
    more than one mask may exist for a image, as 160_{0,1,9}.png
    during training, randomly load one 
    """
    fp = str(fp)
    pil_load = lambda fn: np.array(Image.open(fn))
    # fp = Path(fp)
    fn = fp.split('/')[-1].split('_')[0]
    fn_dir = '/'.join(fp.split('/')[:-1])
    fp_by_doc = lambda i : '/'.join([fn_dir, str('%s_%d.png'%(fn, i))])

    if osp.exists(fp_by_doc(9)):
        return pil_load(fp)
    else:
        fp_candidates = [fp_by_doc(i) for i in (0,1)]
        if np.random.random() > 0.5: fp_candidates = fp_candidates[::-1]
        for cur_fp in fp_candidates:
            if osp.exists(cur_fp):
                return pil_load(cur_fp)
    return pil_load(fp)


pil_load = lambda fn: np.array(Image.open(str(fn)))
get_s_ix = lambda fn: int(fn.stem.split('/')[0])

def mkdir(path):
    # credit goes to YY
    # 判别路径不为空
    path = str(path)
    if path != '':
        # 去除首空格
        path = path.rstrip(' \t\r\n\0')
        if '~' in path:
            path = os.path.expanduser(path)
        # 判别是否存在路径,如果不存在则创建
        if not os.path.exists(path):
            os.makedirs(path)



view2permute = {'saggital': (0, 1, 2),
                'coronal': (2, 0, 1),
                'axial': (1, 2, 0)
                }

# normally z axis should be in first dimension
view2axis = {'saggital': 'xzy',
            'coronal': 'yzx',
            'axial': 'zyx',
            None: None,
            }
# nii dataset always assume xyz dimension order
axis_order_map = {'xzy' : (0, 2, 1), 
                  'zyx': (2, 1, 0), 
                  'yzx': (1, 2, 0),
                  None: None
}

axis_reorder_map = {'xzy' : (0, 2, 1),
                    'zyx' : (2, 1, 0),
                    'yzx' : (2, 0, 1),
                    None: None}

# xyz-> xcyz -> x=0,c=1,y=2,z=3
axis_reorder_map4d = {'xzy' : (0, 1, 3, 2), #xczy xcyz
                      'zyx' : (3, 1, 2, 0), #zcyx xcyz
                      'yzx' : (3, 1, 0, 2)} #yczx xcyz
# image and mask are already saved as nii

def affine_in_sitk_obj(sitk_obj):
    spacing = np.array(sitk_obj.GetSpacing(), dtype = np.float32).reshape(1, 3)
    for i in range(2): spacing[0, i] *= -1 # LPS+ > RAS+, only xy are reversed
    origin = np.array(sitk_obj.GetOrigin(), dtype = np.float32)
    direction = np.array(sitk_obj.GetDirection())
    affine_3x3 = direction.reshape(3, 3) * spacing
    # image_3d = image_3d.transpose(2, 1, 0)
    affine_mat_raw = np.eye(4, dtype = float )
    affine_mat_raw[:3,:3] = affine_3x3
    # for i in range(3): affine_matrix[i, i]  = spacing[i]
    for i in range(3): affine_mat_raw[i, -1] = origin[i]
    return affine_mat_raw

class IO4Nii(object):

    """
    
    Nibabel images always use RAS+ output coordinates, where 
    Right, anterior and superior direction in the patient's perspective is regarded as positive.
    reference: https://nipy.org/nibabel/coordinate_systems.html
    

    If Anatomical Orientation Type (0010,2210) is absent or has a value of BIPED, 
    the x-axis is increasing to the left hand side of the patient. 
    The y-axis is increasing to the posterior side of the patient. 
    The z-axis is increasing toward the head of the patient.
    This is LPS+ coordinates system. 
    reference: https://dicom.innolitics.com/ciods/ct-image/general-series/00102210

    Normally, 0010,2210 tag is absent in the dicom of human subjects, 
    which means that most dicom uses LPS+ coordinates system. 
    When conversion to nifti, we need to transform LPS+ to RAS+ 
    by setting the pixel spacing on axial/xy plane to negative. 

    """
    @staticmethod
    def read(img_nii_fp, verbose = True, axis_order = None, dtype = None, 
             row_first = True):
 
        assert axis_order in list(axis_order_map)

        try:
            if verbose: print('NIBABEL')
            nii_obj = nb.load(img_nii_fp)
            affine_mat_raw = nii_obj.affine.astype(np.float32)
            image_3d_raw = nii_obj.get_fdata()
            # permute_order = axis_order_map[axis_order]
            # image_3d_raw = image_3d_raw.transpose(permute_order)
            # axis_order = None

        except EOFError or OSError:
            # print('[Error] IO4Nii.read ', img_nii_fp)
            print('[IO4Nii] sitk read', img_nii_fp)
            sitk_obj = sitk.ReadImage(img_nii_fp)
            image_3d_raw = sitk.GetArrayFromImage(sitk_obj)
            image_3d_raw = np.transpose(image_3d_raw, (2,1,0))
            affine_mat_raw = affine_in_sitk_obj(sitk_obj)

        # In the patient-centered (world) coordinate system, where x pointing from right to left, y from anterior to posterior, z from feet to head
        # the first 3 rows of the affine matrix are base vectors/directions for xyz axis of the tensor.  
        # cosine 0 degree = 1, cosine 90 degree = 0, cosine 180 degree = -1
        # for instance, the 1st row of (1, 0, 0) mean the x-axis of the tensor is the same as the world coord system. 
        # while the 3st row of (0, 0, -1) mean the z-axis of the tensor is the x-axis reversed of the world coord system.

        # Transform all 
        # [[ 0.          0.         -0.93945312  0.        ]
        #  [ 0.         -0.93945312  0.          0.        ]
        #  [-1.          0.          0.          0.        ]
        #  [ 0.          0.          0.          1.        ]]

        # To, both affine_matrix and image_tensor
        # [[-0.93945312  0.          0.          0.        ]
        #  [ 0.         -0.93945312  0.          0.        ]
        #  [ 0.          0.         -1.0         0.        ]
        #  [ 0.          0.          0.          1.        ]]

        # between axis order
        xyz_dim_index = list(np.argmax(np.abs(affine_mat_raw[:3, :3]), axis= 0 if row_first else 1))
        image_3d_xyz = np.transpose(image_3d_raw, axes = xyz_dim_index)
        affine_matrix = np.copy(affine_mat_raw)
        for i, d in enumerate(xyz_dim_index): 
            if row_first:
                affine_matrix[i, i], affine_matrix[d, i] = affine_mat_raw[d, i], affine_mat_raw[i, i]
            else:
                affine_matrix[i, i], affine_matrix[i, d] = affine_mat_raw[i, d], affine_mat_raw[i, i]
        affine_matrix = np.round(affine_matrix, decimals = 6)

        # within axis direction
        xyz_sign = [int(np.sign(affine_mat_raw[i,i] * (-1 if i <2 else 1))) for i in range(3)]
        xyz_sign = [a if a!= 0 else 1 for a in xyz_sign]
        image_3d_xyz = image_3d_xyz[::xyz_sign[0], ::xyz_sign[1], ::xyz_sign[2]]
        if verbose: 
            print('\n\n[IO4Nii] affine raw\n', affine_mat_raw)
            print('[IO4Nii] affine xyz\n', affine_matrix)
            print('[IO4Nii] xyz dim index', xyz_dim_index)
            print('[IO4Nii] xyz sign', xyz_sign)
            print_tensor('[IO4Nii] image raw', image_3d_raw)
            print_tensor('[IO4Nii] image xyz', image_3d_xyz)

        permute_order = axis_order_map[axis_order]
        if permute_order is None:image_3d = image_3d_xyz
        else:  image_3d = image_3d_xyz.transpose(permute_order)
        if dtype is not None:
            image_3d = image_3d.astype(dtype)
        return image_3d, affine_matrix

    @staticmethod
    def read_ww(img_nii_fp, axis_order = 'zyx',  
                ww = 400, wc = 50, is_uint8 = True, verbose = True):
        
        image_new, affine_matrix = IO4Nii.read(img_nii_fp, 
                                                axis_order = axis_order ,
                                                verbose = verbose)
        if isinstance(ww, int) and isinstance(wc, int):
            image_new = adjust_ww_wl(image_new, ww, wc, is_uint8)
        return image_new, affine_matrix

    @staticmethod
    def read_shape_xyz(img_nii_fp, verbose = False):
        nii_obj = nb.load(img_nii_fp)
        image_shape_xyz = nii_obj.header.get_data_shape()
        # image_shape_zyx = image_shape_xyz[::-1]
        if verbose: print(image_shape_xyz)
        return image_shape_xyz

    @staticmethod
    def write(mask_3d, store_root, file_name, affine_matrix = None, nii_obj= None,
                is_compress = True, axis_order = None):
        """
        must transform the input tensor to xyz, which then can be properly saved.
        and affine matrix are diagonal 
        将输入图像存储为nii
        输入维度是z, r=y, c=x
        输出维度是x=c, y=r, z
        :param mask_3d:
        :param store_root:
        :param file_name:
        :param nii_obj:
        :return:
        """
        extension = 'nii.gz' if is_compress else 'nii'
        permute_order = axis_reorder_map[axis_order]
        if permute_order is not None:
            mask_3d = mask_3d.transpose(permute_order)
        # mask_3d = mask_3d[::-1,::-1,:] # be cautious to uncomment this line
        store_path = osp.join(store_root, '.'.join([file_name, extension]))
        if nii_obj is None:
            if affine_matrix is None: 
                affine_matrix = np.eye(4)
                for i in range(2): affine_matrix[i, i] = -1
            xyz_dim_index = list(np.argmax(np.abs(affine_matrix[:3, :3]), axis= 1))
            xyz_sign = [int(np.sign(affine_matrix[d,i] * (-1 if i <2 else 1))) for i, d in enumerate(xyz_dim_index)]
            # print(affine_matrix, x_col_sign, y_row_sign)
            nb_ojb = nb.Nifti1Image(mask_3d[::xyz_sign[0], ::xyz_sign[1], ::xyz_sign[2]], affine_matrix)
        else:
            nb_ojb = nb.Nifti1Image(mask_3d, nii_obj.affine, nii_obj.header)

        nb.save(nb_ojb, store_path)
        return store_path

    @staticmethod
    def write4d(mask_4d, store_root, file_name, affine_matrix = None, nii_obj= None,
                is_compress = True, axis_order = 'zyx'):
        """
        将输入图像存储为nii 
        输入维度是z, r=y, c=x
        输出维度是x=c, y=r, z
        :param mask_4d: assume the channel dim is the last
        :param store_root:
        :param file_name:
        :param nii_obj:
        :return:
        """
        extension = 'nii.gz' if is_compress else 'nii'
        permute_order = axis_reorder_map[axis_order]
        if permute_order is not None:
            mask_4d = mask_4d.transpose(permute_order + (3, ))
        # mask_3d = mask_3d[::-1,::-1,:] # be cautious to uncomment this line
        store_path = osp.join(store_root, '.'.join([file_name, extension]))
        if nii_obj is None:
            if affine_matrix is None: affine_matrix = np.eye(4) 
            x_col_sign = int(np.sign(-1 * affine_matrix[0,0]))
            y_row_sign = int(np.sign(-1 * affine_matrix[1,1]))
            # print(affine_matrix)
            nb_ojb = nb.Nifti1Image(mask_4d[::x_col_sign, ::y_row_sign, ...], affine_matrix)
        else:
            nb_ojb = nb.Nifti1Image(mask_4d, nii_obj.affine, nii_obj.header)

        nb.save(nb_ojb, store_path)
        return store_path
        
# @staticmethod
def adjust_ww_wl(image, ww = -600, wc = 1600, is_uint8 = True, force2postive = True):
    """
    image is forced to be positive to stay compatible with the following 
    image processing operations using cv2, e.g. padding, elastic transformation and rotation. 
    调整图像得窗宽窗位
    :param image: 3D图像
    :param ww: 窗宽
    :param wc: 窗位
    :return: 调整窗宽窗位后的图像
    """
    min_hu = wc - (ww/2)
    max_hu = wc + (ww/2)
    new_image = np.clip(image, min_hu, max_hu)#np.copy(image)
    if is_uint8:
        new_image -= min_hu
        new_image = np.array(new_image / ww * 255., dtype = np.uint8)
    
    if force2postive and min_hu < 0:
        new_image -= min_hu
    
    return new_image


class ImageDrawerDHW(object):
    """
    start with a image tensor, such as CT volume in D, H, W
    give an index of slice and nb_channels, return a subset of slices centered on that index

    index from the first dimension
    put the indexing dim to the last dim as channels 
    """
    def __init__(self, image_tensor, nb_channels = 3, dim0_to_last = True, skip_step = 1,
                fp = '', verbose = False) -> None:

        self.image = image_tensor
        self.nb_channels = nb_channels
        self.dim0_to_last = dim0_to_last
        self.skip_step = skip_step
        self.fp = fp
        self.verbose = verbose
    
    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, index):
        assert isinstance(index, int)
        assert (index >= 0) and (index < len(self)), print_tensor(f'fp"{self.fp} ix: {index}', self.image )
        pre_nb = int((self.nb_channels - 0.001) // 2)
        post_nb = self.nb_channels - pre_nb - 1
        tg_ix = index
        nb_slices = len(self)
        # tg_ix = 398
        pre_ixs = [max(tg_ix - (pre_nb - i) * self.skip_step , 0) for i in range(pre_nb)]
        post_ixs = [min(tg_ix + (i + 1) * self.skip_step, nb_slices - 1) for i in range(post_nb)]
        ixs = pre_ixs + [tg_ix] + post_ixs
        ip_slices = self.image[ixs, ...]
        if self.dim0_to_last: ip_slices = np.moveaxis(ip_slices, 0, -1)
        return ip_slices, ixs
