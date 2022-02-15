import os, sys, json
import argparse
import pandas as pd
from mmcv.runner import wrap_fp16_model
from mmdet.apis.inference import init_detector

from mmdet.apis.inference_neo import *
from mmdet.datasets.transform4med.io4med import *
from mmdet.datasets.custom_seg import ClassifierPerformanceBinary
from mmdet.datasets.transform4med.load_dicom import affine_matrix_sitk

from lungmask import mask as LungMask

# CLASSES = ('bg', 'hv', 'pv')
CLASSES = ('covid', 'severe')
AGE_MAP = {35: 1, 45: 2, 55: 3, 65: 4, 75: 5, 85: 6}
SEX_MAP = {'F': 0, 'M': 1, 'A': 2, 'O': 2, 'N': 2}

from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for STOIC2021')

    '''
    data_rt = 'data/STOIC2021Round1',
    fn2imglist = 'stoic2021_case_info_split.csv',
    prefix_dir = 'processed', 
    split = 'test', file_suffix = '.nii'
    '''
    parser.add_argument('--repo-rt',
                        help='repo/code root of mmseg',
                        default= 'git/stoic2021/work_dirs',
                        # required=True,
                        type=str)
    parser.add_argument('--model-name',
                        help='model name',
                        default= 'resnset_s32c16em2_stoic2kcv05_256x224x224_2cls_agecode',
                        # required=True,
                        type=str)
    parser.add_argument('--best-weight',
                        help='weigth checkpoint with best auc',
                        default= 'best_auc_epoch_6',
                        # required=True,
                        type=str)

    parser.add_argument('--data-rt',
                        help="data rt ",
                        default='git/stoic2021/data/STOIC2021Round1',
                        type=str
                        )
    parser.add_argument('--fn2imglist',
                        help=" filename to image list ",
                        default='stoic2021_case_info_split.csv',
                        type=str
                        )
    parser.add_argument('--prefix-dir',
                        help="prefix dir ",
                        default='processed',
                        type=str
                        )
    parser.add_argument('--split',
                        help="data split ",
                        default='test',
                        type=str
                        )
    parser.add_argument('--cv-fold',
                        help="which fold to infer ",
                        default=0,
                        type=int
                        )
    parser.add_argument('--gpu-ix',
                        help="gpu-ix, just 1",
                        default= 0,
                        type=int
                        )              
    parser.add_argument('--fold-ix',
                        help="which fold of the data to be infered. Using parallel inference for speedup",
                        default=None,
                        type=int
                        )            
    parser.add_argument('--num-fold',
                    help="number of folds to distribute. Using parallel inference for speedup",
                    default=4,
                    type=int
                    )                         
    parser.add_argument('--is-run',
                        help="flag for testing",
                        action='store_true'
                    )        
    parser.add_argument('--seed',
                        help="random seed for spliting the dataset",
                        default= 42,
                        type=int
                    )        
    parser.add_argument('--run_pid_ixs',
                    help="select pids to run",
                    type=int,
                    nargs='+',
                    # type = list,
                )      
    parser.add_argument('--verbose',
                    action='store_true',
                    help="if verbose",
                )                                    
    args = parser.parse_args()
    for a in vars(args):  print(f'{a}\t{getattr(args, a)}')
    return args


def infer_loop(cfg, 
        nii_save_dir = '',  
        pid2niifp_map = dict(),
        is_test = False,
        fold_ix_str = '0@1', 
        ):
    CLASSES = ('covid', 'severe')
    # model configuration prepare
    git_rt = Path(cfg.repo_rt)
    # model_name = 'fcn_hr18_449x449_40k_pancreas_neg100_dl'; process_func = process_1_case_2d
    class_predictor = CovidSeverePredictor(git_rt, cfg.model_name, 
                                            cfg.best_weight, device = f'cuda:{cfg.gpu_ix}')
    # sys.exit('debug')
    result_by_pids = []

    print('\nCases to infer : %d \n' %len(pid2niifp_map))
    pcount = 0

    # infer by model
    for pid, fp_dict in pid2niifp_map.items(): #[1:2]
        img_nii_fp = fp_dict['image']
        gt_nii_fp = fp_dict.get('label', None)
        # ipdb.set_trace()   
        print(f'\n[INFER] {pcount} {img_nii_fp}, gt {gt_nii_fp}')
        target_cls, age_num, sex_num, pid = stoic_class_from_fname(img_nii_fp)
        this_holder = {'pid': pid, 'infer_time_avg': '', 
                'age':age_num, 'sex': sex_num, 
                'gt_cls': np.array(target_cls)}    

        if is_test: continue 

        sitk_image = sitk.ReadImage(img_nii_fp)
        cls_prob_1x2 = class_predictor(sitk_image, age = age_num)
        pred_catg = cls_prob_1x2 > 0.5
        print_tensor(f'\t cls prob', cls_prob_1x2)

        for i, prob in enumerate(cls_prob_1x2):
            this_holder[f'cls{i}_gt_catg'] = target_cls[i]
            this_holder[f'cls{i}_gt_name'] = CLASSES[i] if target_cls[i] else 'BG'
            this_holder[f'cls{i}_pred_catg'] = pred_catg[i]
            this_holder[f'cls{i}_pred_name'] = CLASSES[i] if pred_catg[i] else 'BG'
            this_holder[f'cls{i}_pred_prob'] = float(prob)
        result_by_pids.append(this_holder)
        pcount += 1

    eval_results = {}
    for cls_i in range(2):
        cls_name = CLASSES[cls_i]
        # ipdb.set_trace()
        cls_results = np.vstack([a[f'cls{cls_i}_pred_prob'] for a in result_by_pids])
        gt_labels = np.vstack([a[f'cls{cls_i}_gt_catg'] for a in result_by_pids])
        eval_results_i = ClassifierPerformanceBinary(gt_labels, cls_results, cls_name = cls_name) 
        eval_results.update(eval_results_i)

    for k, v in eval_results.items():
        if isinstance(v, (np.ndarray, torch.Tensor, list)):
            print(f'\n{k} : \n', v)
        else:
            print(f'\n{k} : {v:.2f}')

    result_tb = pd.DataFrame(result_by_pids)
    result_fps = {'scores' : osp.join(nii_save_dir, f'aug_inference_{fold_ix_str}.csv'),
                'summary': osp.join(nii_save_dir, f'inference_summary_{fold_ix_str}.csv'),
                'model_dir': model_store_dir}

    result_tb.to_csv(result_fps['scores'], index = False)
    summary_tb = result_tb.describe()
    summary_tb.to_csv(result_fps['summary'], index = False)
    print(result_tb.describe())

    return result_fps


class CovidSeverePredictor:
    def __init__(self, model_dir, model_name, best_weight, device = 'cuda:0', cfg = None):
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.cfg = cfg
        self.extend3axis_mm = (16, 12, 8)
        self.target_spacings = [(0.9, 0.9, 0.9), (1.0, 1.0, 1.0), (1.1, 1.1, 1.1)]
        
        if not model_name.endswith('nan'):
            self.cls_model = self._create_cls_model(model_dir, model_name, best_weight)  # create model
            self.num_classes = len(self.cls_model.CLASSES)
        
        self.lung_segmentor = LungMask.get_model('unet','R231')

    def _create_cls_model(self, model_dir, model_name, best_weight):
        """ filenames of model weigth and config are the same """
        config_file = f'{model_dir}/{model_name}/{model_name}.py'
        weight_file = f'{model_dir}/{model_name}/{best_weight}.pth'
        print(config_file)
        print(weight_file)
        model = init_detector(str(config_file), str(weight_file), device=self.device)
        wrap_fp16_model(model)
        return model

    def __call__(self, img_sitk, age = None):
        
        affine_matrix = affine_matrix_sitk(img_sitk)
        # print('\taffine \n', affine_matrix)
        img_3d_origin = sitk.GetArrayFromImage(img_sitk) #.transpose(2, 1, 0)
        # with Timer(print_tmpl='\tInferLung {:.3f} seconds'): 

        with suppress_stdout():
            lung_3d_origin = LungMask.apply(img_sitk, batch_size=32, 
                                            model = self.lung_segmentor,) #.transpose(2, 1, 0) # xyz

        img_ori_size = img_3d_origin.shape
        extend3axis_pixel = [int(m/abs(affine_matrix[i, i])) for 
                                i, m in enumerate(self.extend3axis_mm)]
        lung_coords = np.where(lung_3d_origin > 0)
        lung_slicer = tuple([slice(max(min(lung_coords[i]) - ext, 0), 
                                   min(max(lung_coords[i]) + ext, img_ori_size[i])) 
                                 for i, ext in enumerate(extend3axis_pixel)])
        img_3d_lung = img_3d_origin[lung_slicer]
        mask_3d_lung = lung_3d_origin[lung_slicer]

        lung_lengths = [max(lung_coords[a]) - min(lung_coords[a]) for a in range(3)]
        print(f'\t lung length {lung_lengths} Lung shape extend z{extend3axis_pixel}', img_3d_lung.shape)
        covid_severe_prob = self.inference(img_3d_lung, affine_matrix, 
                                            lung_mask = mask_3d_lung, age = age)
        return covid_severe_prob

    def inference(self, image_3d, affine_matrix, lung_mask = None, age = None):

        raw_arr_xyz = image_3d.transpose(2, 1, 0)
        # model_rt = Path(self.cls_model_path).parent
        # IO4Nii.write(raw_arr_xyz, model_rt, 'test_shape_order',
        #              self.imageset.affine_matrix, axis_order=None)
       
        # seg_prob_whole = np.zeros_like(raw_arr_xyz, dtype=np.float32)
        # seg_count_whole = np.zeros_like(raw_arr_xyz, dtype=np.uint8)

        prob_1x2 = tta_classify_1by1(self.cls_model, raw_arr_xyz, affine = affine_matrix, 
                                        target_spacings = self.target_spacings, 
                                        flip_directions=[None, 'diagonal'], 
                                        age = age)
        # seg_prob_chns = seg_prob_chns.float().numpy()
        # seg_results.append(seg_prob_chns)
        return prob_1x2

def stoic_class_from_fname(abs_path):
    fname = str(abs_path).split('/')[-1].split('.')[0]
    pid, age, sex, covid, severe = fname.split('_')
    age = age[3:]
    sex = sex[3:]
    covid = int(covid[5:])
    severe = int(severe[6:])
    age_num = AGE_MAP[int(age[1:-1]) if len(age) > 2 else int(age)]
    sex_num = SEX_MAP.get(sex, 2)
    target_cls = [covid, severe]
    return target_cls, age_num, sex_num, pid
    

def load_sample_list(data_rt = 'data/STOIC2021Round1',
                    fn2imglist = 'stoic2021_case_info_split.csv',
                    prefix_dir = 'processed', 
                    split = 'test', 
                    cv_fold = 0, 
                    file_suffix = '.nii'):
    
    img_list_fp = os.path.join(data_rt, fn2imglist)
    assert osp.exists(img_list_fp)
    data_tb = pd.read_csv(img_list_fp)
    if split is not None:
        select_mask = data_tb['split']!= cv_fold if split == 'train' \
                            else data_tb['split'] == cv_fold
    else:
        select_mask = ~pd.isna(data_tb['split'])
    
    data_tb = data_tb.loc[select_mask, :]

    sample_list = [
        {'pid' : data_tb.loc[i, 'PatientID'], 
        'image': os.path.join(data_rt, prefix_dir, data_tb.loc[i, 'img_path'] + file_suffix), 
        'label': os.path.join(data_rt, prefix_dir, data_tb.loc[i, 'img_path'] + file_suffix)}
                        for i in data_tb.index]

    pid2niifp_map = {cinfo['pid']: cinfo  for cinfo in sample_list}
    # pathpairs_orderd = sorted(sample_list, key = lambda x: x['pid'])
    print(f'[RawCT] {len(pid2niifp_map)} samples')
    return pid2niifp_map

pid_in_path_rb = lambda x : x.split('_')[0]

def sample_list2run(pid2niifp_map, run_pid_ixs = None,
                    num_fold = 4, fold_ix = None):
    run_pids = list(pid2niifp_map)
    run_pids = sorted(run_pids)#[:20]
    total_pid = len(run_pids)
    print(f'Dataset contains {total_pid} pid in total')
    if fold_ix is not None:
        num_per_fold = int(np.ceil(total_pid / num_fold)) #+ (1 if total_pid % num_fold > 0 else 0)
        if fold_ix * num_per_fold < total_pid:
            start = fold_ix * num_per_fold
            end = (fold_ix + 1) * num_per_fold #if fold_ix != num_fold else total_pid
            if fold_ix == num_fold -1: end = total_pid
            print(f'[Fold{fold_ix}/{num_fold}]: start {start} end {end}')
            run_pids = run_pids[start: end]
        else: run_pids = []
    # save_string_list(data_rt/'entire_sub_dir_list.txt', [str(v) for v in pid2niifp_map.values()])
    # [print(i, p) for i, p in enumerate(run_pids)]
    # print(f'Fold{fold_ix}/{num_fold}:  total pids {len(run_pids)}, first 2 {run_pids[:2]}, last 2 {run_pids[-2:]} \n')
    run_pids = [run_pids[i] for i in run_pid_ixs] if bool(run_pid_ixs) else run_pids
    print(f'Fold{fold_ix}/{num_fold}:  total pids {len(run_pids)}, first 2 {run_pids[:2]}, last 2 {run_pids[-2:]} \n')
    
    pid2niifp_map = {p:pid2niifp_map[p] for p in run_pids}
    return pid2niifp_map

if __name__ == '__main__':

    # data path preparation 
    cfg = parse_args()
    # print(cfg.model, cfg.run_pid_ixs)
    dataset_name = 'stoic2021'

    pid2niifp_map = load_sample_list(cfg.data_rt, cfg.fn2imglist, 
                                    cfg.prefix_dir, cfg.split, 
                                    cv_fold=cfg.cv_fold,
                                    )
    
    pid2niifp_map = sample_list2run(pid2niifp_map, 
                                    run_pid_ixs=cfg.run_pid_ixs, 
                                    num_fold=cfg.num_fold, 
                                    fold_ix=cfg.fold_ix)

    fold_ix_str = f'{cfg.fold_ix}@{cfg.num_fold}'

    git_rt = Path(cfg.repo_rt)
    model_store_dir = git_rt/cfg.model_name
    store_name =  str(f'visual_{dataset_name}_{cfg.split}_nii')
    nii_save_dir = model_store_dir/store_name
    nii_save_dir.mkdir(parents = True, exist_ok=True)

    infer_loop(cfg, 
        nii_save_dir = nii_save_dir, 
        pid2niifp_map=pid2niifp_map,
        fold_ix_str = fold_ix_str,
        # is_test=True
        )