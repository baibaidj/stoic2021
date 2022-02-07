import numpy as np
from .metric_cls import support, precision_recall_f1
from .accuracy import accuracy
from ..utils import print_tensor
import pdb

from sklearn.metrics import (accuracy_score, roc_auc_score, recall_score, 
                            confusion_matrix, make_scorer, roc_curve)

def tn(y_true, y_pred):
    result = confusion_matrix(y_true, y_pred)

    return result[0,0]/np.sum(result[0,:])

def fp(y_true, y_pred):
    result = confusion_matrix(y_true, y_pred)
    return result[0, 1] / np.sum(result[0, :])

def fn(y_true, y_pred):
    result = confusion_matrix(y_true, y_pred)
    return result[1, 0] / np.sum(result[1, :])

def tp(y_true, y_pred):
    result = confusion_matrix(y_true, y_pred)
    return result[1, 1] / np.sum(result[1, :])


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



def ClassifierPerformanceBinary(gt_labels, pred_probs, threshold = 0.5, 
                                cls_name = '0', 
                                model_name = 'cnext'):

    """
    Args:
        gt_labels: nx1, \in {0, 1}
        pred_probs: nx1, \in [0, 1]

    Returns:
        [type]: [description]
    """

    # pdb.set_trace()
    # print_tensor('[Eva] class pred', pred_probs)
    # print_tensor('[Eva] class gt', gt_labels)
    num_imgs = len(pred_probs)
    assert len(gt_labels) == num_imgs, 'dataset testing results should '\
        'be of the same length as gt_labels.'


    pred_catg = pred_probs > threshold
    acc = accuracy_score(gt_labels, pred_catg )
    recall = recall_score(gt_labels, pred_catg)
    specifity = tn(gt_labels, pred_catg)
    auc = roc_auc_score(gt_labels, pred_probs)
    print(f'\n{model_name} {cls_name} threshold {threshold} acc{acc:.4f} recall{recall:.4f} ppv{specifity:.4f} auc{auc:.4f}')

    eval_results = {}
    eval_results[f'{cls_name}_acc'] = acc
    eval_results[f'{cls_name}_recall'] = recall
    eval_results[f'{cls_name}_ppv'] = specifity
    eval_results[f'{cls_name}_auc'] = auc

    return eval_results
        