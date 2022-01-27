import os.path as osp
from copy import deepcopy

import torch, mmcv
from mmcv.runner import HOOKS, Hook
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.log_buffer import LogBuffer

from mmdet.core.evaluation import DistEvalHookMed, EvalHookMed


@HOOKS.register_module()
class SWAHook(Hook):
    r"""SWA Object Detection Hook.

        This hook works together with SWA training config files to train
        SWA object detectors <https://arxiv.org/abs/2012.12645>.

        Args:
            swa_eval (bool): Whether to evaluate the swa model.
                Defaults to True.
            eval_hook (Hook): Hook class that contains evaluation functions.
                Defaults to None.
            swa_interval (int): The epoch interval to perform swa
    """

    def __init__(self, swa_eval=True, eval_hook=None, swa_interval=1):
        if not isinstance(swa_eval, bool):
            raise TypeError('swa_eval must be a bool, but got'
                            f'{type(swa_eval)}')
        if swa_eval:
            if not isinstance(eval_hook, EvalHookMed) and \
               not isinstance(eval_hook, DistEvalHookMed):
                raise TypeError('eval_hook must be either a EvalHook or a '
                                'DistEvalHook when swa_eval = True, but got'
                                f'{type(eval_hook)}')
        self.swa_eval = swa_eval
        self.eval_hook = eval_hook
        self.swa_interval = swa_interval

    def before_run(self, runner):
        """Construct the averaged model which will keep track of the running
        averages of the parameters of the model."""
        model = runner.model
        self.model = AveragedModel(model)

        self.meta = runner.meta
        if self.meta is None:
            self.meta = dict()
            self.meta.setdefault('hook_msgs', dict())
        if isinstance(self.meta, dict) and 'hook_msgs' not in self.meta:
            self.meta.setdefault('hook_msgs', dict())
        self.log_buffer = LogBuffer()

    def after_train_epoch(self, runner):
        """Update the parameters of the averaged model, save and evaluate the
        updated averaged model."""
        model = runner.model
        # Whether to perform swa
        if (runner.epoch + 1) % self.swa_interval == 0:
            swa_flag = True
        else:
            swa_flag = False
        # update the parameters of the averaged model
        if swa_flag:
            self.model.update_parameters(model)

            # save the swa model
            runner.logger.info(
                f'Saving swa model at swa-training {runner.epoch + 1} epoch')
            filename = 'swa_runavg_model_{}.pth'.format(runner.epoch + 1)
            filepath = osp.join(runner.work_dir, filename)
            optimizer = runner.optimizer
            self.meta['hook_msgs']['last_ckpt'] = filepath
            save_checkpoint(
                self.model.module,
                filepath,
                optimizer=optimizer,
                meta=self.meta)

        # evaluate the swa model
        if self.swa_eval and swa_flag:
            self.work_dir = runner.work_dir
            self.rank = runner.rank
            self.epoch = runner.epoch
            self.iter = runner.iter
            self.logger = runner.logger
            self.optimizer = runner.optimizer
            self.meta['hook_msgs']['last_ckpt'] = filename
            self.eval_hook.after_train_epoch(self)
            for name, val in self.log_buffer.output.items():
                name = 'swa_' + name
                runner.log_buffer.output[name] = val
            runner.log_buffer.ready = True
            self.log_buffer.clear()

    def after_run(self, runner):
        # since BN layers in the backbone are frozen,
        # we do not need to update the BN for the swa model
        # Update bn statistics for the swa_model at the end

        pass

    def before_epoch(self, runner):
        pass

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'swa_avg_latest.pth')
            mmcv.symlink(filename, dst_file)



class AveragedModel(torch.nn.Module):
    r"""Implements averaged model for Stochastic Weight Averaging (SWA).
    AveragedModel class creates a copy of the provided model on the device
    and allows to compute running averages of the parameters of the model.
    Args:
        model (torch.nn.Module): model to use with SWA
        device (torch.device, optional): if provided, the averaged model
            will be stored on the device. Defaults to None.
        avg_fn (function, optional): the averaging function used to update
            parameters; the function must take in the current value of the
            AveragedModel parameter, the current value of model
            parameter and the number of models already averaged; if None,
            equally weighted average is used. Defaults to None.
    """

    def __init__(self, model, device=None, avg_fn=None):
        super(AveragedModel, self).__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:

            def avg_fn(averaged_model_parameter, model_parameter,
                       num_averaged):
                return averaged_model_parameter + (
                    model_parameter - averaged_model_parameter) / (
                        num_averaged + 1)

        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(
                    self.avg_fn(p_swa.detach(), p_model_,
                                self.n_averaged.to(device)))
        self.n_averaged += 1


# import torch
# from mmcv.runner import HOOKS, Hook, ClosureHook, Runner
# inspiration https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/

# from torch.optim.swa_utils import AveragedModel, SWALR
# from torch.optim.lr_scheduler import CosineAnnealingLR

# @HOOKS.register_module()
# class SWAOptimizer(Hook):

#     def __init__(self, swa_start = 10, swa_lr = 0.05, ):
#         self.swa_model = None
#         self.swa_start = swa_start
#         self.swa_lr = swa_lr
#         pass

#     def before_run(self, runner):
        
#         pass

#     def after_run(self, runner):
#         pass

#     def before_epoch(self, runner : Runner):
#         if runner.epoch == self.swa_start:
#             self.swa_model = AveragedModel(runner.model)
#             self.swa_scheduler = SWALR(runner.optimizer, swa_lr=self.swa_lr)
#             # self.

#     def after_epoch(self, runner):
#         if runner.epoch > self.swa_start:
#             self.swa_model.update_parameters(runner.model)
#             self.swa_scheduler.step()

# @HOOKS.register_module()
# class SWALrUpdaterHook():


# from torch.optim.swa_utils import AveragedModel, SWALR
# from torch.optim.lr_scheduler import CosineAnnealingLR

# loader, optimizer, model, loss_fn = ...
# # fine-tuning pretrained model with SWA
# # create a new hook to store and update swa_model.
# # using CosineRestartLrUpdaterHook or CyclicLrUpdaterHook 
# swa_model = AveragedModel(model) # apply during initialization 
# scheduler = CosineAnnealingLR(optimizer, T_max=100) # change to lR update hook by mmcv 
# swa_start = 5
# swa_scheduler = SWALR(optimizer, swa_lr=0.05) 
# # change to lR update hook by mmcv: CosineRestartLrUpdaterHook or 

# for epoch in range(100):
#       for input, target in loader:
#           optimizer.zero_grad()
#           loss_fn(model(input), target).backward()
#           optimizer.step()

#       if epoch > swa_start: # on epoch end
#           swa_model.update_parameters(model)
#           swa_scheduler.step()
#       else:
#           scheduler.step()

# # Update bn statistics for the swa_model at the end
# # torch.optim.swa_utils.update_bn(loader, swa_model)
# # Use swa_model to make predictions on test data 
# # preds = swa_model(test_input)