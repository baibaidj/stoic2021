from .custom_seg import CustomDatasetMonai, classifier_performance
from .transform4med.io4med import (
    os, osp, load_string_list, np, 
    Path, convert_label, print_tensor)
import pandas as pd

from .builder import DATASETS


@DATASETS.register_module()
class STOIC21Dataset(CustomDatasetMonai):
    """Pneumonia dataset.

    The ``img_suffix`` is fixed to '_leftImg8bewdcfit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    CLASSES = ('covid', 'severe')
    
    def __init__(self, *args, cv_fold = 0 , 
                prefix_dir = 'processed', file_extension = '.nii', 
                **kwargs):

        self.cv_fold = cv_fold
        self.prefix_dir = prefix_dir
        self.file_extension = file_extension

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
    


    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        # if self.gt_seg_maps is not None:
        #     return self.gt_seg_maps
        if getattr(self, 'gt_seg_maps', None) is not None:
            return getattr(self, 'gt_seg_maps', None)

        # attr_in_transform = lambda x, trans: [f for f in trans if hasattr(f, x)]
        label_mapping = self.pipeline.transforms[1].label_mapping
        value4outlier = getattr(self.pipeline.transforms[1], 'value4outlier', 0)
        # num_slices = self.pipeline.transforms[0].num_slice
        gt_seg_maps = {}
        for i, img_info in enumerate(self.img_infos):
            fp = Path(img_info['gt_semantic_seg'])
            subdir = str(fp.stem) #view2axis[self.view_channel]
            img_cls = min(int(subdir.split('_')[-1]), self.num_class_cls) - 1 # remove lumps class
            # img_full, af_mat = IO4Nii.read(fp, verbose=True, axis_order= None, dtype=np.uint8)
            img = self.reader.read(fp)
            img_full, meta_data = self.reader.get_data(img)
            af_mat = meta_data['original_affine']
            if i < 3: 
                print_tensor(f'\n[GT] {subdir} {self.view_channel}', img_full) #
                print('[GT] af matrix\n', af_mat)
            gt_seg_map  = np.array(img_full, dtype = np.uint8)
            # print('Eval select transform', len(result_dicts))
            # pdb.set_trace()
            gt_seg_maps.setdefault(subdir, {'gix':[], 'pix' :[], 'affine':None, 
                                            'gt': [], 'gt_cls':[img_cls], 'ixs': []})
            gt_seg_maps[subdir]['gix'].append(i)
            gt_seg_maps[subdir]['affine'] = af_mat

            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255
            if label_mapping is not None: gt_seg_map = convert_label(gt_seg_map, label_mapping, 
                                                        value4outlier = value4outlier)

            if i < 3: print_tensor(f'gt-convertlabel {label_mapping}' , gt_seg_map) #
            # print('eval, mask', gt_seg_map.shape, gt_seg_map.min(), gt_seg_map.max())
            gt_seg_maps[subdir]['gt'].append(gt_seg_map)
        self.gt_seg_maps = gt_seg_maps
        return gt_seg_maps

    def get_cls_gt_labels(self):
        gt_seg_maps = self.get_gt_seg_maps()
        cls_gt_labels = []
        for pid, info in gt_seg_maps.items():
            cls_gt_labels.extend(info['gt_cls'])
        return cls_gt_labels
        

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
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
        allowed_metrics = ['mIoU']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gt_by_pids = self.get_gt_seg_maps()

        num_classes_cls = len(self.CLASSES)

        i = 0
        result_by_pids = []
        for i, (pid, info) in enumerate(gt_by_pids.items()): 
            *_, cls_probs = results[info['gix'][0]]
            this_holder = {'pid': pid} 
            pred_prob, pred_catg, gt_catg = cls_probs.max(), int(cls_probs.argmax()), info['gt_cls'][0]
            this_holder.update({'cls_gt_catg': gt_catg, 'cls_gt_name': self.CLASSES[gt_catg], 
                                'cls_pred_catg': pred_catg, 'cls_pred_name' : self.CLASSES[pred_catg], 
                                'cls_pred_prob': pred_prob, 'cls_pred_raw': cls_probs})

            # if i < 2: print_tensor(f'[Metric]{pid} pred {np.unique(pred_tensor)}', pred_tensor)
            # if i < 2: print_tensor(f'[Metric]{pid} gt {np.unique(gt_tensor)}', gt_tensor)
            if i < 2: print(this_holder)
            result_by_pids.append(this_holder)
            i += 1
        
        # pdb.set_trace()
        cls_results = np.vstack([a.pop('cls_pred_raw') for a in result_by_pids])
        gt_labels = np.vstack([a['cls_gt_catg'] for a in result_by_pids])
        eval_results = classifier_performance(cls_results, gt_labels)
        return eval_results, result_by_pids