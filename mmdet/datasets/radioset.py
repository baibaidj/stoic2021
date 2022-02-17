from .custom_seg import CustomDatasetMonai, ClassifierPerformanceBinary
from .transform4med.io4med import (
    os, osp, load_string_list, np, 
    Path, convert_label, print_tensor)
import pandas as pd
import ipdb
from .builder import DATASETS


@DATASETS.register_module()
class STOIC21Dataset(CustomDatasetMonai):
    """Pneumonia dataset.

    The ``img_suffix`` is fixed to '_leftImg8bewdcfit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    CLASSES = ('covid', 'severe')
    AGE_MAP = {35: 1, 45: 2, 55: 3, 65: 4, 75: 5, 85: 6}
    SEX_MAP = {'F': 0, 'M': 1, 'A': 2, 'O': 2, 'N': 2}
    
    def __init__(self, *args, cv_fold = 0 , target_class = (0, 1),
                prefix_dir = 'processed', file_extension = '.nii', 
                **kwargs):

        self.cv_fold = cv_fold
        self.prefix_dir = prefix_dir
        self.file_extension = file_extension
        self.target_class = target_class

        super(STOIC21Dataset, self).__init__(*args, **kwargs)
        self.gt_seg_maps = None
        self.flag = np.ones(len(self), dtype=np.uint8)

    def _img_list2dataset(self, data_folder:str, **kwags):
        """
        
        return 
            file_list : [{'image' : img_path, 'label' : label_path}, ...]
        """
        # a = [print(self.map_key(k)) for k in keys]
        js_fp = os.path.join(data_folder, self.fn2imglist)
        if not osp.exists(js_fp): return []
        if js_fp.endswith('txt'):
            image_fns = load_string_list(js_fp)
        elif js_fp.endswith('csv'):
            case_tb = pd.read_csv(js_fp)
            select_mask = case_tb['split']!= self.cv_fold if self.split == 'train' \
                         else case_tb['split'] == self.cv_fold
            image_fns =  case_tb.loc[select_mask, 'img_path']

        pid2pathpairs = []
        for ifn  in image_fns:
            cid = ifn.split('_')[0]
            img_fp = f'{self.img_dir}/{self.prefix_dir}/{ifn}{self.file_extension}'
            this_pair = {'cid': cid, 'img': img_fp}
            pid2pathpairs.append(this_pair)
        pathpairs_orderd = sorted(pid2pathpairs, key = lambda x: x['cid'])
        print(f'[RawCT] {len(pathpairs_orderd)} samples')
        return pathpairs_orderd

    def get_cls_gt_labels(self):
        """Get ground truth segmentation maps for evaluation."""
        # if self.gt_seg_maps is not None:
        #     return self.gt_seg_maps
        if getattr(self, 'gt_cls_labels', None) is not None:
            return getattr(self, 'gt_cls_labels', None)

        # num_slices = self.pipeline.transforms[0].num_slice
        gt_cls_labels = {}
        for i, img_info in enumerate(self.img_infos):
            abs_fp = Path(img_info['img'])
            target_cls, age_num, sex_num, subdir = self.stoic_class_from_fname(abs_fp)
            # img_cls = min(int(subdir.split('_')[-1]), self.num_class_cls) - 1 # remove lumps class
            # print('Eval select transform', len(result_dicts))
            # pdb.set_trace()
            if self.target_class is not None: 
                target_cls = [target_cls[i] for i in self.target_class]

            gt_cls_labels.setdefault(subdir, 
                        {'gix':i, 'pix' :0, 'age':age_num, 'sex': sex_num, 
                        'gt_cls': np.array(target_cls)})

        self.gt_cls_labels = gt_cls_labels
        return gt_cls_labels

    def evaluate(self, results, metric='auc', logger=None, return_casewise = False, **kwargs):
        """Evaluate the dataset. will be called by core/evalutation/eval_hooks.py

        Args:
            results (list[tuple(seg_prob, aux_prob, cls_prob)]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['auc', 'recall', 'ppv', 'acc', 'severe_auc', 'covid_auc']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gtcls_by_pids = self.get_cls_gt_labels()

        i = 0
        result_by_pids = []
        num_pred_cls = len(self.CLASSES)
        for ip, (pid, info) in enumerate(gtcls_by_pids.items()):  
            pred_prob = results[info['gix']]
            this_holder = {'pid': pid}
            # ipdb.set_trace()
            if pred_prob.size > len(self.target_class):
                pred_catg = [np.argmax(pred_prob)]
                pred_prob = [pred_prob[i + 1] for i in self.target_class]
            else:
                pred_catg = pred_prob > 0.5
            gt_catg = info['gt_cls']
            for i, prob in enumerate(pred_prob):
                this_holder[f'cls{i}_gt_catg'] = gt_catg[i]
                this_holder[f'cls{i}_gt_name'] = self.CLASSES[i] if gt_catg[i] else 'BG'
                this_holder[f'cls{i}_pred_catg'] = pred_catg[i]
                this_holder[f'cls{i}_pred_name'] = self.CLASSES[i] if pred_catg[i] else 'BG'
                this_holder[f'cls{i}_pred_prob'] = float(prob)
            if ip < 2: 
                num_pred_cls = len(pred_prob)
                print('\n', this_holder)
            result_by_pids.append(this_holder)

        eval_results = {}
        for cls_i in range(num_pred_cls):
            cls_name = self.CLASSES[cls_i]
            # ipdb.set_trace()
            cls_results = np.vstack([a[f'cls{cls_i}_pred_prob'] for a in result_by_pids])
            gt_labels = np.vstack([a[f'cls{cls_i}_gt_catg'] for a in result_by_pids])
            eval_results_i = ClassifierPerformanceBinary(gt_labels, cls_results, cls_name = cls_name)
            eval_results.update(eval_results_i)

        eval_results['mean_auc'] =  np.mean([eval_results[f'{self.CLASSES[cls_i]}_auc'] for 
                                        cls_i in range(num_pred_cls)])
        if return_casewise: 
            return eval_results, result_by_pids
        else: 
            return eval_results

    def stoic_class_from_fname(self, abs_path):
        fname = str(abs_path).split('/')[-1].split('.')[0]
        pid, age, sex, covid, severe = fname.split('_')
        age = age[3:]
        sex = sex[3:]
        covid = int(covid[5:])
        severe = int(severe[6:])
        age_num = self.AGE_MAP[int(age[1:-1]) if len(age) > 2 else int(age)]
        sex_num = self.SEX_MAP.get(sex, 2)
        target_cls = [covid, severe]
        return target_cls, age_num, sex_num, pid




@DATASETS.register_module()
class AllCTDataset(CustomDatasetMonai):
    """Pneumonia dataset.

    The ``img_suffix`` is fixed to '_leftImg8bewdcfit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    CLASSES = ('bg', 'fg')
    def __init__(self, *args, 
                cv_fold = 0 , target_class = (0, 1),
                prefix_dir = 'processed', file_extension = '.nii', 
                **kwargs):
        self.cv_fold = cv_fold
        self.prefix_dir = prefix_dir
        self.file_extension = file_extension
        self.target_class = target_class
        super(AllCTDataset, self).__init__(*args,**kwargs)

        self.gt_seg_maps = None
        self.flag = np.ones(len(self), dtype=np.uint8)


    
    def _img_list2dataset(self, data_folder:str, **kwags):
        """
        
        return 
            file_list : [{'image' : img_path, 'label' : label_path}, ...]
        """
        # a = [print(self.map_key(k)) for k in keys]
        js_fp = os.path.join(data_folder, self.fn2imglist)
        if not osp.exists(js_fp): return []
        if js_fp.endswith('txt'):
            image_fns = load_string_list(js_fp)
        elif js_fp.endswith('csv'):
            case_tb = pd.read_csv(js_fp)
            image_fns = case_tb['img_path']

        # pid2pathpairs = []
        # for ifp  in image_fns:
        #     cid = ifp.split(os.sep)[-1].split('.')[0]
        #     this_pair = {'cid': cid, 'img': ifp}
        #     pid2pathpairs.append(this_pair)
        # pathpairs_orderd = sorted(pid2pathpairs, key = lambda x: x['cid'])
        # print(f'[RawCT] {len(pathpairs_orderd)} samples')
        # return pathpairs_orderd

        pid2pathpairs = []
        for ifn  in image_fns:
            cid = ifn.split('_')[0]
            img_fp = f'{self.img_dir}/{self.prefix_dir}/{ifn}{self.file_extension}'
            this_pair = {'cid': cid, 'img': img_fp}
            pid2pathpairs.append(this_pair)
        pathpairs_orderd = sorted(pid2pathpairs, key = lambda x: x['cid'])
        print(f'[RawCT] {len(pathpairs_orderd)} samples')
        return pathpairs_orderd