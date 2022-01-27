
import numpy as np
from mmcv import Timer
import pandas as pd
import multiprocessing
import timeit, json, random, os
from monai.transforms.utils_ import MaskFgLocator3d
from pathlib import Path
import os.path as osp 
from monai.utils import fall_back_tuple, ensure_tuple_rep

def print_dict(your_dict):
    print(json.dumps(your_dict, indent=4, sort_keys=True))

import fastremap


def save2json(obj, data_rt, filename, indent=4, sort_keys=True):
    assert 'json' in filename
    fp = osp.join(data_rt, filename)
    with open(fp, 'w', encoding='utf8') as f:
        json.dump(obj, f, sort_keys=sort_keys, ensure_ascii=False, indent=indent)
    return fp

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


def format_labels(labels, in_place):
    if in_place:
        labels = fastremap.asfortranarray(labels)
    else:
        labels = np.copy(labels, order='F')

    if labels.dtype == bool:
        labels = labels.view(np.uint8)

    original_shape = labels.shape

    while labels.ndim < 3:
        labels = labels[..., np.newaxis ]

    while labels.ndim > 3:
        if labels.shape[-1] == 1:
            labels = labels[..., 0]
        else:
            raise ValueError(
            "Input labels may be no more than three non-trivial dimensions. Got: {}".format(
              original_shape
            )
          )

    return labels

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

def clock(func):
    def clocked(*args, **kwargs):
        t0 = timeit.default_timer()
        result = func(*args,  **kwargs)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        # arg_str = ', '.join(repr(arg) for arg in args)
        print('%s takes %0.4fs' % (name, elapsed))
        return result
    return clocked

def multi_process_wrapper(num_thread = 10, verbose = True):
    """
    execute a function in multi processes, this wrapper makes several assumptions. 
    1. the first argument to the function is an iterables which will be splited into different processes as a guide. 
    2. the return objects of the function, if it is a dict, then its values must be list or tuple. 
    3. the function must be a stand-alone object instead of a class method that has to be called with a self handle. 

    """
    cpu_count = multiprocessing.cpu_count()
    if num_thread <= 0 or num_thread > cpu_count:
        num_thread = cpu_count

    is_dict = lambda x : isinstance(x, dict)
    def update_dict(holder, inputs):
        assert isinstance(holder, dict)
        assert isinstance(inputs, dict)
        
        for k, v in inputs.items():
            holder.setdefault(k, []).extend(v)
        # return holder

    def force2list(x):
        if isinstance(x, list):
            return x
        else:
            return list(x)
    
    def real_decorator(func_obj):
        def inter_logic(iterables, *args, **kwargs):
            sample_idx = sorted(list(iterables)) if is_dict(iterables) else list(range(len(iterables)))
            sample_num = len(sample_idx)
            # func_name = 'new_func_%d'%i
            # class eval(func_name)(func_obj):
            #     def __init__(self, a = None):
            #         self.a = a
            #         super.__init__()
            # func = eval(func_name)()
            func = func_obj()
            if num_thread > 1:  #
                per_part = len(sample_idx) // num_thread + 1
                pool = multiprocessing.Pool(processes=num_thread)
                process_list = []
                for i in range(num_thread):
                    # if i in [18]:
                    start = int(i * per_part)
                    stop = int((i + 1) * per_part)
                    stop = min(stop, sample_num)
                    if verbose: print('thread=%d, start=%d, stop=%d' % (i, start, stop))
                    if not is_dict(iterables):
                        inter_iterables = [iterables[k] for k in sample_idx[start:stop]]
                    else:
                        inter_iterables = {k : iterables[k] for k in sample_idx[start:stop]}
                    this_proc = pool.apply_async(func, args=(inter_iterables, ) + args, kwds=kwargs)
                    process_list.append(this_proc)
                    # print('here')
                pool.close()
                pool.join()
                
                final_return = []
                for pix, proc in enumerate(process_list):
                    return_objs = proc.get()
                    if isinstance(return_objs, type(None)): return
                    for rix, reo in enumerate(return_objs):
                        if pix == 0:
                            if is_dict(reo):
                                final_return.append(dict()) # assume that the keys are list
                            else:
                                final_return.append(list())
                        if is_dict(reo):
                            update_dict(final_return[rix], reo)
                        else:
                            final_return[i].extend(force2list(reo))
            else:
                final_return = func(iterables, *args, **kwargs)
            return final_return
        return inter_logic
    return real_decorator

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


def run_parralel_1by1(main_func, arg_list):
    num_run = len(arg_list)
    pool = multiprocessing.Pool(num_run)
    process_list = []
    for i, arg in enumerate(arg_list):
        this_proc = pool.apply_async(main_func, args = arg)
        process_list.append(this_proc)

    pool.close()
    pool.join()
    final_return = []
    for pix, proc in enumerate(process_list):
        return_objs = proc.get()
        final_return.append(return_objs)
    return tuple(final_return)


def split_cases_labeldir(cids, src_label_dir, dst_label_dir, split_ratio = 0.8, seed = 42, 
                            is_test = True):
    num_cids = len(cids)
    split_point = int(num_cids * split_ratio)
    random.seed(42)
    shuffle_index = list(range(num_cids))
    random.shuffle(shuffle_index)
    train_index = shuffle_index[: split_point]
    test_index = shuffle_index[split_point:]

    split_sets = {'labelsTr': train_index, 'labelsTs':test_index}
    
    for sset, split_ixs in split_sets.items():
        dst_set_dir = osp.join(dst_label_dir, sset)
        mkdir(dst_set_dir)
        for ix in split_ixs:
            cid = cids[ix]
            cid_tries = [cid, cid.split('_')[0]]
            src_label_fp = None
            for c in cid_tries:
                label_fp = osp.join(src_label_dir, f'case_{c}.nii.gz')
                if osp.exists(label_fp): src_label_fp = label_fp; break
            if src_label_fp is None: print(f'no label path exists for {cid}'); continue
            
            transfer_cmd = f'ln -s {src_label_fp} {dst_set_dir}'
            print(transfer_cmd)
            if not is_test: os.system(transfer_cmd)
    return split_sets

from contextlib import contextmanager
import sys, os
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":

    proj_rt= Path('/root/data')
    taskid = 'Task006'
    info_fn = 'case_info_records_skeleton.csv'

    src_label_dir = proj_rt/'labelsAll_0404'
    dst_label_dir = proj_rt/taskid
    task_tb = pd.read_csv(proj_rt/info_fn)
    task_tb.set_index('caseid', inplace = True)
    task_tb.sort_index(inplace = True)
    cids = list(task_tb.index)
    split_sets = split_cases_labeldir(cids, src_label_dir, dst_label_dir, is_test=False)
    task_tb['split'] = None
    for sset, split_ixs in split_sets.items():
        for i in split_ixs: 
            # print(i)
            task_tb.iloc[int(i), -1] = sset

    print(task_tb.head())
    task_tb.to_csv(proj_rt/f'split_{info_fn}')
