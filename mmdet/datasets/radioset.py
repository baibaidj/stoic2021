from .custom_seg import CustomDatasetMonai
from .transform4med.io4med import os, osp, load_string_list, np
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
        super(STOIC21Dataset, self).__init__(*args, **kwargs)

        self.cv_fold = cv_fold
        self.prefix_dir = prefix_dir
        self.file_extension = file_extension
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
    