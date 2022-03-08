
import collections
import pydicom
import SimpleITK as sitk
import numpy as np
import os, ipdb, copy
import os.path as osp
import pandas as pd
from pathlib import Path

def store_sqeuence_1by1(d, s3, name = 'spacing_dim'):
    assert isinstance(d, dict) and isinstance(s3, collections.abc.Sequence)
    for i, s in enumerate(s3): d[f'{name}{i}'] = s

def get_sqeuence_1by1(d, num = 3, name = 'spacing_dim'):
    assert isinstance(d, dict)
    return tuple([d[f'{name}{i}'] for i in range(num)])


def load_dicom(path, method=0, verb = False):
    path = str(path)
    volumes = []
    if method == 0:
        reader = sitk.ImageSeriesReader()
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
        for series_uid in series_IDs:
            dicom_names = reader.GetGDCMSeriesFileNames(path, series_uid)
            if verb: print(f'\t load dicom series {series_uid}')
            reader.SetFileNames(dicom_names)
            volume = reader.Execute()
            volumes.append(volume)
        return volumes, series_IDs
    else:
        slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(
                slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(
                slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness

        volume = np.stack([s.pixel_array for s in slices])

        pixel_presentation = slices[0][(0x0028, 0x0103)].value
        volume = volume.astype(
            np.uint16) if pixel_presentation == 0 else volume.astype(np.int16)

        volume = sitk.GetImageFromArray(volume)
        voxel_size = [float(slices[0][(0x0028, 0x0030)].value[0]), float(slices[0][(0x0028, 0x0030)].value[1]),
                      slice_thickness]
        volume.SetSpacing(voxel_size)
        volumes.append(volume)
        return volumes, []
        
def load_image_and_info(img_s_dir, series_info = {}, worker = 99, verbose = True):

    series_info['origin'] = None
    series_info['direction'] = None
    store_sqeuence_1by1(series_info, [None] *3, name = 'spacing_dim')
    store_sqeuence_1by1(series_info, [None] *3, name = 'size_dim')
    img_datas, series_infos = [], []
    img_data_series, series_names = load_dicom(img_s_dir, verb=verbose)
    # series_label_infos.append(series_info)
    num_series = 0 if img_data_series is None else len(img_data_series)

    for i in range(num_series):
        img_data = img_data_series[i]
        img_spacing_xyz = img_data.GetSpacing()
        img_shape_xyz = img_data.GetSize() #[::-1]
        series_info_i = copy.deepcopy(series_info)
        # TODO: add other important acquisition parameters, including manufacturer, reconstruction kernel, kpv, mA
        store_sqeuence_1by1(series_info_i, img_spacing_xyz, name = 'spacing_dim')
        store_sqeuence_1by1(series_info_i, img_shape_xyz, name = 'size_dim')
        series_info_i['origin'] = img_data.GetOrigin()
        series_info_i['direction'] = img_data.GetDirection()
        series_info_i['series_name'] = series_names[i]

        img_spacing_xyz = get_sqeuence_1by1(series_info_i, name = 'spacing_dim')
        img_shape_xyz = get_sqeuence_1by1(series_info_i, name = 'size_dim')
        if verbose: print(f'\t\t[Worker{worker}]CT image {series_names[i]}', img_shape_xyz, img_spacing_xyz)
        img_datas.append(img_data)
        series_infos.append(series_info_i)
        # series_label_infos.append(series_info)
    return img_datas, series_infos

def series2num(phase, p2d_map = {'V': 0, 'A': 1, 'PV': 2, 'PS': 3},):
    d = p2d_map.get(phase, 9)
    return d
bind_pid_series2fn = lambda cid, sname : f'case_{cid}_000{series2num(sname)}.nii.gz'

bind_pid_series = lambda cid, sname : f'case_{cid}_{sname}.nii.gz'

def _dicom2nii1case(img_s_dir, dst_img_dir,  worker = 99, verb = False):
    # read DICOM data
    case_id = '_'.join(img_s_dir.split(os.sep)[-2:])
    series_info = {'cid': case_id}
    img_datas, series_infos = load_image_and_info(img_s_dir, series_info, worker, verb)
    # store image 
    for img_data, series_info_i in zip(img_datas, series_infos):
        series_name = series_info_i['series_name']
        img_fn = bind_pid_series(case_id, series_name)
        save_img_fp = osp.join(dst_img_dir , img_fn)
        print(f'\t [dicom2nii] {case_id} to {img_fn}')
        if not osp.exists(save_img_fp):
            sitk.WriteImage(img_data, save_img_fp)

    return series_info


def Dicom2NiiLoop(img_dirs, store_dir, prcs_ix = 99):
    
    case_info_list = []
    for i, img_dir in enumerate(img_dirs):
        case_info = _dicom2nii1case(img_dir, store_dir, worker = prcs_ix)
        case_info_list.append(case_info)

    return case_info_list


def affine_matrix_sitk(sitk_image, row_first = True):
    spacing = np.array(sitk_image.GetSpacing(), dtype=np.float32).reshape(1, 3)
    spacing[0, :2] *= -1
    origin = np.array(sitk_image.GetOrigin(), dtype=np.float32)
    direction = np.array(sitk_image.GetDirection())
    affine_3x3 = direction.reshape(3, 3) * spacing
    # image_3d = image_3d.transpose(2, 1, 0)
    affine_mat_raw = np.eye(4, dtype = float )
    affine_mat_raw[:3,:3] = affine_3x3
    # for i in range(3): affine_matrix[i, i]  = spacing[i]
    for i in range(3): affine_mat_raw[i, -1] = origin[i]
    xyz_dim_index = list(np.argmax(np.abs(affine_mat_raw[:3, :3]), axis= 0 if row_first else 1))
    affine_matrix = np.copy(affine_mat_raw)
    for i, d in enumerate(xyz_dim_index):
        if row_first:
            affine_matrix[i, i], affine_matrix[d, i] = affine_mat_raw[d, i], affine_mat_raw[i, i]
        else:
            affine_matrix[i, i], affine_matrix[i, d] = affine_mat_raw[i, d], affine_mat_raw[i, i]
    affine_matrix = np.round(affine_matrix, decimals = 6)

    return affine_matrix

if __name__ == '__main__':

    set_name = 'leg'
    pn_rt = Path(f'/Users/monolith/Desktop/leg_seg') 
    # {set_name}_dicom
    raw_set_dir = pn_rt/f'{set_name}_dicom'
    store_set_dir = pn_rt/f'{set_name}_nii'

    case_dcm_dirs = []
    for subr, subd, subf in os.walk(raw_set_dir):
        # ipdb.set_trace()
        if len(subf) > 0:
            print('check', subf)
            if 'dcm' in subf[0]:
                case_dcm_dirs.append(subr)
    
    case_info_list, *_ = Dicom2NiiLoop(case_dcm_dirs, store_set_dir, 
                                    )
    
    case_tb = pd.DataFrame(case_info_list)
    case_tb.to_csv(store_set_dir/f'case_info_tb_{set_name}.csv')

    
