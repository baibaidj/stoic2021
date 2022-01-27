
import warnings, sys
import torch, json
import torch, mmcv
import pandas as pd
from copy import deepcopy
from functools import reduce

from mmcv.utils import print_log
from torch.utils.data import Dataset

from .transform4med.io4med import *
from .builder import DATASETS
from .pipelines import Compose
from multiprocessing.pool import ThreadPool
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union

from monai.transforms import Randomizable, Transform, apply_transform
from monai.utils import min_version, optional_import
from monai.data.image_reader import NibabelReader
from mmdet.core.evaluation.metric_custom import *

if TYPE_CHECKING:
    from tqdm import tqdm
    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")

def decide_cid_in_fn(fn):
    fn_stem = fn.split('.')[0]
    stem_chunks = fn_stem.split('_')
    if len(stem_chunks) == 2:
        stem_chunks.append('0000')
    return '_'.join(stem_chunks[1:])

def add_series_chunk_safe(fn):
    fn_chunks = fn.split('_')
    if len(fn_chunks) == 2: return fn.replace('.nii.gz', '_0000.nii.gz')
    else: return fn


@DATASETS.register_module()
class CustomDatasetMonai(Dataset):
    """Custom dataset for semantic segmentation.

    An example of file structure is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 ann_dir=None,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 sample_rate = 1.0,
                 view_channel = None,
                 verbose = False,
                 keysmap = {'img': 'image', 'gt_semantic_seg': 'label'},
                 keys = ('img', 'gt_semantic_seg'),
                 exclude_pids = None,
                 fn2imglist = 'dataset.json',
                 fn_spliter = ['_', 1],
                 target_classes = None,

                 ):

        self.keysmap = keysmap
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir if data_root is None else osp.join(data_root, img_dir)
        # self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        # self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.sample_rate = sample_rate
        self.view_channel = view_channel
        self.keys = keys
        self.verbose = verbose
        self.exclude_pids = exclude_pids
        self.fn2imglist  = fn2imglist
        self.fn_spliter = fn_spliter
        self.reader = NibabelReader()
        self.target_classes = target_classes

        self.num_class_cls = len(self.CLASSES)
        print(f'[Dataset] contains {self.num_class_cls}: {self.CLASSES}' )
        # load annotations
        self.img_infos = self._img_list2dataset(self.img_dir, keys = self.keys, mode = self.split)
        self.img_infos = self._sample_img_data(self.img_infos, self.sample_rate)

        # print(self.img_infos)

    def map_key(self, key):
        # mm2monai_map = self.keysmap#{'img' : 'image', 'gt_semantic_seg' : 'label'}
        return self.keysmap.get(key, key)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def _load_sample_list(self, data_folder, mode):
        
        js_fp = os.path.join(data_folder, self.fn2imglist)
        assert osp.exists(js_fp)
        if self.fn2imglist.endswith('json'):
            with open(js_fp, 'r') as load_f:
                load_dict = json.load(load_f)
            sample_list = load_dict['training'] if mode == 'train' else load_dict['test']
            # tmp = 'Tr/' if mode == 'train' else 'Ts/'
        elif self.fn2imglist.endswith('csv'):
            data_tb = pd.read_csv(js_fp)
            # 1.3.12.2.1107.5.1.4.60358.30000018031901030776900109333 mask shape not equal tp image
            # data_tb = data_tb.loc[data_tb['instance_class'] == 2, :]
            if self.target_classes is not None:
                data_tb = data_tb.loc[data_tb['instance_class'].isin(self.target_classes), :]
            # data_tb['instance_class'] = data_tb['instance_class'].apply(
            #                             lambda x: min(self.num_class_cls, x))
            print('[Load] instance class unique :\n', data_tb['instance_class'].value_counts())
            sample_list = [{'image': data_tb.loc[i, 'image']+'.nii', 
                            'label': data_tb.loc[i, 'label']+ '.nii'}
                             for i in data_tb.index]
        else:
            NotImplementedError

        return sample_list

    def _img_list2dataset(self, data_folder:str, keys = ('img', 'gt_semantic_seg'), mode = 'train '):
        """
        
        return 
            file_list : [{'image' : img_path, 'label' : label_path}, ...]
        """
        # a = [print(self.map_key(k)) for k in keys]
        assert self.map_key(keys[0]) == 'image' and self.map_key(keys[1]) == 'label'
        sample_list = self._load_sample_list(data_folder, mode)

        # img_ids = []
        pid_in_fp_ky = lambda p: p.split(os.sep)[-1].split(self.fn_spliter[0])[self.fn_spliter[1]]
        pid2pathpairs = []
        for ix, path_pair in enumerate(sample_list):
            img_ikey, lab_ikey = [keys[i] for i in range(2)]
            img_fkey, lab_fkey = [self.map_key(k) for k in (img_ikey, lab_ikey)]
            fname = path_pair[img_fkey][2:]
            img_fn = add_series_chunk_safe(fname)
            lab_fn = path_pair[lab_fkey][2:]
            cid = pid_in_fp_ky(img_fn.split('.nii.gz')[0])
            if ix < 3: print('Check cid', cid)
            if self.exclude_pids and (cid in self.exclude_pids): 
                print('Exclude ', cid)
                continue
            input_path_pair = {'cid': cid}
            input_path_pair[img_ikey] = os.path.join(data_folder, img_fn)
            input_path_pair[lab_ikey] = os.path.join(data_folder, lab_fn)

            if len(keys) > 2:
                for k in keys[2:]:
                    this_fn = lab_fn.replace(lab_fkey+'s', k)
                    input_path_pair[k] = osp.join(data_folder, this_fn)
            if ix < 2: print(f'[mode{mode}] Check path pair {ix}', input_path_pair)
            pid2pathpairs.append(input_path_pair)

        pathpairs_orderd = sorted(pid2pathpairs, key = lambda x: x['cid'])
        return pathpairs_orderd

    def _sample_img_data(self, fp_list, sample_rate = 1.0, rand_seed = 42):
        
        if sample_rate< 1.0:
            np.random.seed(rand_seed)
            nb_total = len(fp_list)
            sample_indexes = np.random.choice(np.arange(nb_total), int(nb_total * sample_rate), 
                                            replace=False)
            sample_indexes = sorted(sample_indexes)                                
            fp_list = [fp_list[a] for a in sample_indexes]
        return fp_list

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        results = self.img_infos[idx]
        # results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        results = self.img_infos[idx]
        # results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        return class_names

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
        if self.CLASSES is None:
            num_classes_cls = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_by_pids]))
        else:
            num_classes_cls = len(self.CLASSES)
        
        has_cls_result = results[0][-1] is not None
        num_classes_seg = 2 if has_cls_result else num_classes_cls

        # all_acc, acc, iou, dice2d, dice3d, recall3d, precision3d = [list() for _ in range(7)]
        # msg_by_pid = '\n\tpid\t\tacc\trecall\tprecision\tdice2d\tdice3d\n'
        # f24 = lambda x: '%0.4f'%f24
        # prog_bar = mmcv.ProgressBar(len(gt_seg_maps))

        # verbose = True
        # reorganize the pred and gt by pids
        # find all the gix that belong to individual pids. 
        i = 0
        pred_seg_list = [results[info['gix'][0]][0] for pid, info in gt_by_pids.items()]
        gt_seg_list = [info['gt'][0]  for pid, info in gt_by_pids.items()]
        seg_metric_detail, seg_metric_cls = segmentation_performance(pred_seg_list, gt_seg_list, num_classes_seg)
        seg_mean_results = organize_seg_performance(seg_metric_cls, self.CLASSES[:num_classes_seg]) # 
        
        for k, seg_tb in seg_metric_detail.items():
            print(f'[SegMetric]{k}\n', seg_tb[:5])

        result_by_pids = []
        for i, (pid, info) in enumerate(gt_by_pids.items()): 
            *_, cls_probs = results[info['gix'][0]]
            this_holder = {'pid': pid, 'seg_dice': seg_metric_detail['dice'][i, 1:], 
                            'seg_recall': seg_metric_detail['recall'][i, 1:], 
                            'seg_preciesion': seg_metric_detail['precision'][i, 1:]} 
            if has_cls_result:
                pred_prob, pred_catg, gt_catg = cls_probs.max(), int(cls_probs.argmax()), info['gt_cls'][0]
                this_holder.update({'cls_gt_catg': gt_catg, 'cls_gt_name': self.CLASSES[gt_catg], 
                                    'cls_pred_catg': pred_catg, 'cls_pred_name' : self.CLASSES[pred_catg], 
                                    'cls_pred_prob': pred_prob, 'cls_pred_raw': cls_probs})

            # if i < 2: print_tensor(f'[Metric]{pid} pred {np.unique(pred_tensor)}', pred_tensor)
            # if i < 2: print_tensor(f'[Metric]{pid} gt {np.unique(gt_tensor)}', gt_tensor)
            if i < 2 and has_cls_result: print(this_holder)
            result_by_pids.append(this_holder)
            i += 1
        if not has_cls_result : return eval_results, result_by_pids
        
        # pdb.set_trace()
        cls_results = np.vstack([a.pop('cls_pred_raw') for a in result_by_pids])
        gt_labels = np.vstack([a['cls_gt_catg'] for a in result_by_pids])

        eval_results = classifier_performance(cls_results, gt_labels)

        return eval_results, result_by_pids

    def _compute_metric_seg(self):


        pass


class CacheDatasetMonai(CustomDatasetMonai):
    """
    Dataset with cache mechanism that can load data and cache deterministic transforms' result during training.

    By caching the results of non-random preprocessing transforms, it accelerates the training data pipeline.
    If the requested data is not in the cache, all transforms will run normally
    (see also :py:class:`monai.data.dataset.Dataset`).

    Users can set the cache rate or number of items to cache.
    It is recommended to experiment with different `cache_num` or `cache_rate` to identify the best training speed.

    To improve the caching efficiency, please always put as many as possible non-random transforms
    before the randomized ones when composing the chain of transforms.

    For example, if the transform is a `Compose` of::

        transforms = Compose([
            LoadImaged(),
            AddChanneld(),
            Spacingd(),
            Orientationd(),
            ScaleIntensityRanged(),
            RandCropByPosNegLabeld(),
            ToTensord()
        ])

    when `transforms` is used in a multi-epoch training pipeline, before the first training epoch,
    this dataset will cache the results up to ``ScaleIntensityRanged``, as
    all non-random transforms `LoadImaged`, `AddChanneld`, `Spacingd`, `Orientationd`, `ScaleIntensityRanged`
    can be cached. During training, the dataset will load the cached results and run
    ``RandCropByPosNegLabeld`` and ``ToTensord``, as ``RandCropByPosNegLabeld`` is a randomized transform
    and the outcome not cached.
    """

    def __init__(
        self,
        *args,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: Optional[int] = None,
        progress: bool = True,
        **kwargs
    ) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: transforms to execute operations on input data.
            cache_num: number of items to be cached. Default is `sys.maxsize`.
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            cache_rate: percentage of cached data in total, default is 1.0 (cache all).
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            num_workers: the number of worker processes to use.
                If num_workers is None then the number returned by os.cpu_count() is used.
            progress: whether to display a progress bar.
        """
        # if not isinstance(transform, Compose):
        #     transform = Compose(transform)
        self.progress = progress
        super(CacheDatasetMonai, self).__init__(*args, **kwargs)
        self.cache_num = min(int(cache_num), int(len(self) * cache_rate), len(self))
        self.num_workers = num_workers
        if self.num_workers is not None:
            self.num_workers = max(int(self.num_workers), 1)
        self._cache: List = self._fill_cache()

    def _fill_cache(self) -> List:
        if self.cache_num <= 0:
            return []
        if self.progress and not has_tqdm:
            warnings.warn("tqdm is not installed, will not show the caching progress bar.")
        with ThreadPool(self.num_workers) as p:
            if self.progress and has_tqdm:
                return list(
                    tqdm(
                        p.imap(self._load_cache_item, range(self.cache_num)),
                        total=self.cache_num,
                        desc="Loading dataset",
                    )
                )
            return list(p.imap(self._load_cache_item, range(self.cache_num)))

    def _load_cache_item(self, idx: int):
        """
        Args:
            idx: the index of the input data sequence.
        """
        item = self.img_infos[idx]; self.pre_pipeline(item)
        if not isinstance(self.pipeline, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")
        for _transform in self.pipeline.transforms:
            # execute all the deterministic transforms
            if isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                break
            item = apply_transform(_transform, item)
        return item

    def __getitem__(self, index):
        if index >= self.cache_num:
            # no cache for this index, execute all the transforms directly
            return super(CacheDatasetMonai, self).__getitem__(index)
        # load data from cache and execute from the first random transform
        start_run = False
        if self._cache is None:
            self._cache = self._fill_cache()
        data = self._cache[index]
        if not isinstance(self.pipeline, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")
        for _transform in self.pipeline.transforms:
            if start_run or isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                start_run = True
                data = apply_transform(_transform, data)
        return data


def overlap_handler_zaxis(tensor_list, indices_list, stack_axis = 0, verbose = True):
    """
    locate overlap of zaxis for predition of a CT series 
    and take the mean for the overlap region

    tensor_list: [3dtensor_1, 3dtensor_2]

    """
    # assert len(tensor_list)
    is_torch_tensor = isinstance(tensor_list[0], torch.Tensor)
    if is_torch_tensor:
        stack_func = torch.stack
        device = tensor_list[0].device
        dim_arg = 'dim'
    else:
        stack_func = np.stack
        device = None
        dim_arg = 'axis'
    slice_pred_dict = {}
    total_slices = 0
    for tensor1, ixs in zip(tensor_list, indices_list):
        for i, z_ix in enumerate(ixs):
            pred_i = tensor1[i, ...].to(device) if device else tensor1[i, ...] # 3d tensor, 
            # if verbose: print_tensor('\tpred %d' %i, pred_i)
            slice_pred_dict.setdefault(z_ix, []).append(pred_i)
            total_slices += 1
    
    unique_z_ixs = sorted(list(slice_pred_dict))
    if verbose: print('\t total: %d, unique:' %total_slices, len(unique_z_ixs), unique_z_ixs[:3], unique_z_ixs[-3:])
    pred4d = stack_func([stack_func(slice_pred_dict[z], **{dim_arg : -1 }).sum(**{dim_arg : -1 }) / len(slice_pred_dict[z]) 
                        for z in unique_z_ixs], **{dim_arg :stack_axis})
    # pred4d = np.array(pred4d, dtype = np.uint8)
    if verbose: print_tensor('\tpred4d', pred4d)
    return pred4d


def norm_dim_4_torch(tensor, verbose = False, device = 0):
    """
    transform a 5d tensor to 4d tensor assuming the batch-size dim is 1

    """
    if verbose: print_tensor('\tNORM_DIM:', tensor)
    tensor_dim = len(tensor.shape)
    if tensor_dim == 5:
        b, c, d, h, w = tensor.shape
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.permute(0, 2, 1, 3, 4).view(b * d, c, h, w)
        else:
            tensor = tensor.transpose(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
    if tensor_dim == 4:
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.permute(1, 0, 2, 3)
        else:
            tensor = tensor.transpose(1, 0, 2, 3)
    return tensor

def norm_dim_4_np(tensor):
    """
    transform a 5d tensor to 4d tensor assuming the batch-size dim is 1

    """
    if len(tensor.shape) == 5:
        b, c, d, h, w = tensor.shape
        tensor = tensor.transpose(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
    return tensor


denormalize = lambda x: 2.0 * x - 1.0

def logit2mask(seg_logit):
    seg_pred = seg_logit.argmax(dim=1) if seg_logit.shape[1] > 1 else seg_logit.squeeze(1) > 0.5
    seg_pred = seg_pred.int().cpu().numpy()
    return seg_pred