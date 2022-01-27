
from mmcv.utils import print_log
import numpy as np
from .metric_cls import support, precision_recall_f1
from .accuracy import accuracy
from .mean_dice import metric_in_cfsmat_1by1, cfsmat4mask_batched
from collections import OrderedDict
from prettytable import PrettyTable
from ..utils import print_tensor
import pdb


def classifier_performance(cls_results, gt_labels):

    """
    Args:
        cls_results: nxc
        gt_labels: nx1

    Returns:
        [type]: [description]
    """
    metric_options = {'topk': (1, 3)}
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'support']

    # pdb.set_trace()
    eval_results = {}
    # cls_results = np.vstack([a.pop('cls_pred_raw') for a in result_by_pids])
    # gt_labels = np.vstack([a['cls_gt_catg'] for a in result_by_pids])
    print_tensor('[Eva] class pred', cls_results)
    print_tensor('[Eva] class gt', gt_labels)
    num_imgs = len(cls_results)
    assert len(gt_labels) == num_imgs, 'dataset testing results should '\
        'be of the same length as gt_labels.'

    topk = metric_options.get('topk', (1, 3))
    thrs = metric_options.get('thrs', None)
    average_mode = metric_options.get('average_mode', 'macro')

    if 'accuracy' in metrics:
        acc = accuracy(cls_results, gt_labels, topk=topk, thrs=thrs)
        if isinstance(topk, tuple):
            eval_results_ = {
                f'accuracy_top-{k}': a
                for k, a in zip(topk, acc)
            }
        else:
            eval_results_ = {'accuracy': acc}
        if isinstance(thrs, tuple):
            for key, values in eval_results_.items():
                eval_results.update({
                    f'{key}_thr_{thr:.2f}': value.item()
                    for thr, value in zip(thrs, values)
                })
        else:
            eval_results.update(
                {k: v.item()
                    for k, v in eval_results_.items()})

    if 'support' in metrics:
        support_value, cfs_mat = support(
            cls_results, gt_labels, average_mode=average_mode)
        eval_results['cfs_mat'] = cfs_mat.tolist()

    precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
    if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
        precision_recall_f1_values = precision_recall_f1(
            cls_results, gt_labels, average_mode=average_mode, thrs=thrs)
        for key, values in zip(precision_recall_f1_keys,
                                precision_recall_f1_values):
            if key in metrics:
                if isinstance(thrs, tuple):
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value
                        for thr, value in zip(thrs, values)
                    })
                else:
                    eval_results[key] = values

    return eval_results


def segmentation_performance(seg_results, seg_gt_masks, num_classes_seg = 2, ignore_index = 255):
    """

    these metrics include 'iou', 'dice', 'acc', 'all_acc', 'recall', 'precision'
    metric3d: dict {iou: [class0, class1, ...],
                    dice: [class0, class1, ...]}
    Args:
        seg_results: list[Tensor] # Tensor: 4D, 1HWD
        seg_gt_masks: list[np.ndarray] # 3d, HWD
    
    Return:
        dict: {iou: nxc ndarray, 
               dice: nxc ndarray,  
               ...}
    """

    assert len(seg_results) == len(seg_gt_masks)
    metric3d_list = []
    for i, (pred_seg, gt_seg) in enumerate(zip(seg_results, seg_gt_masks)):
        if pred_seg[0].shape != gt_seg.shape: 
            print(f'[EvalSeg] {i} pred {pred_seg[0].shape} not equal to gt {gt_seg.shape} in shape ')
            continue
        cfs_matrix_list = cfsmat4mask_batched(list(pred_seg[0]), list(gt_seg), num_classes_seg, ignore_index)
        metric2ds, metric3d = metric_in_cfsmat_1by1(cfs_matrix_list)
        # by class, by metric 
        metric3d_list.append(metric3d)
    
    metric_keys = list(metric3d_list[0].keys())
    key2metric_detail = {}
    # pdb.set_trace()
    for k in metric_keys: key2metric_detail[k] = np.array([a[k] for a in metric3d_list])  # nxcls
    # cx1
    key2metric_cls = {k: list(np.nanmean(key2metric_detail[k], axis = 0)) for k in metric_keys}
    return key2metric_detail, key2metric_cls


def organize_seg_performance(ret_metrics, class_names, logger=None,):
    num_cls = len(class_names)
    # pdb.set_trace()
    # summary table
    ret_metrics_summary = OrderedDict({ 
        ret_metric: np.round(np.nanmean(ret_metric_value[-num_cls:]) * 100, 2) for  
            ret_metric, ret_metric_value in ret_metrics.items() })

    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
            ret_metric: list(np.round(ret_metric_value[-num_cls:], 2)) for 
                        ret_metric, ret_metric_value in ret_metrics.items()})

    ret_metrics_class.update({'Class': list(class_names)})
    ret_metrics_class.move_to_end('Class', last=False)

    # for logger
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items(): class_table_data.add_column(key, val)

    summary_table_data = PrettyTable() 
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            summary_table_data.add_column(key, [val])
        else:
            summary_table_data.add_column('m' + key, [val])

    print_log('per class results:', logger)
    print_log('\n' + class_table_data.get_string(), logger=logger)
    print_log('Summary:', logger)
    print_log('\n' + summary_table_data.get_string(), logger=logger)

    eval_results = {}
    # each metric dict
    for key, value in ret_metrics_summary.items():
        if key == 'aAcc':
            eval_results[key] = value / 100.0
        else:
            eval_results['m' + key] = value / 100.0

    ret_metrics_class.pop('Class', None)
    for key, value in ret_metrics_class.items():
        eval_results.update({
            key + '.' + str(name): value[idx] / 100.0
            for idx, name in enumerate(class_names)
        })
    return eval_results
        