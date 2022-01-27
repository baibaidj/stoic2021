import os
import os.path as osp
import numpy as np
import nibabel as nb
import SimpleITK as sitk
import pandas as pd
import multiprocessing
from pathlib import Path
import time, json, random, pdb, cc3d
from skimage.measure import regionprops

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

is_dict = lambda x : isinstance(x, dict)

def update_dict(holder, inputs):
    assert isinstance(holder, dict)
    assert isinstance(inputs, dict)
    
    for k, v in inputs.items():
        if isinstance(v, (list, tuple)):
            holder.setdefault(k, []).extend(v)
        else:
            holder[k] = v
    # return holder

def force2list(x):
    if isinstance(x, list):
        return x
    else:
        return list(x)

def run_parralel(func, iterables, *args, num_workers = 1, verb = False, **kwargs):
    cpu_count = multiprocessing.cpu_count()
    if num_workers <= 0 or num_workers > cpu_count:
        num_workers = cpu_count


    sample_idx = sorted(list(iterables)) if is_dict(iterables) else list(range(len(iterables)))
    sample_num = len(sample_idx)
    kwargs['prcs_ix'] = 99
    if verb: print('[MultiProcess] Run parallel: num samples %d'  % sample_num)
    if num_workers > 1:  #
        per_part = sample_num // num_workers + (1 if sample_num % num_workers > 0  else 0)
        pool = multiprocessing.Pool(processes=num_workers)
        process_list = []
        for i in range(num_workers): #num_thread
            # if i in [18]:
            kwargs['prcs_ix'] = i
            start = int(i * per_part)
            stop = int((i + 1) * per_part)
            stop = min(stop, sample_num)
            if verb: print('[MultiProcess]thread=%d, start=%d, stop=%d' % (i, start, stop))
            if is_dict(iterables):
                inter_iterables = {k : iterables[k] for k in sample_idx[start:stop]}
            elif isinstance(iterables, pd.DataFrame):
                inter_iterables = iterables.iloc[start:stop]
            else:
                inter_iterables = [iterables[k] for k in sample_idx[start:stop]]
            this_proc = pool.apply_async(func, args=(inter_iterables, ) + args, kwds=kwargs)
            process_list.append(this_proc)
            # print('here')
        pool.close()
        pool.join()
        
        final_return = []

        if verb: print('[MultiProcess] Gather results')
        for pix, proc in enumerate(process_list):
            if verb: print(f'[MultiProcess] worker {pix}')
            return_objs = proc.get()
            if isinstance(return_objs, type(None)): return
            if not isinstance(return_objs, tuple):
                return_objs = (return_objs, )
            for rix, reo in enumerate(return_objs): 
                
                if pix == 0:
                    if is_dict(reo):
                        final_return.append(dict()) # assume that the keys are list
                    else:
                        final_return.append(list())
                if is_dict(reo):
                    update_dict(final_return[rix], reo)
                else:
                    final_return[rix].extend(force2list(reo))
            if verb: print('[MultiProcess] Done')
    else:
        final_return = func(iterables, *args, **kwargs)
        if not isinstance(final_return, tuple):
            final_return = (final_return, )

    return final_return

