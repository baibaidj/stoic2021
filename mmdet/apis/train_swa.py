
from .train import *
from torch.optim.swa_utils import update_bn as swa_update_bn

def train_detector_swa(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):


    logger = get_root_logger(cfg.log_level)


    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed, persistent_workers = True,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)


    # perform swa training
    # build swa training runner

    from mmdet.core import SWAHook
    logger.info('Start SWA training')
    swa_optimizer = build_optimizer(model, cfg.swa_optimizer)
    swa_runner = build_runner(
        cfg.swa_runner,
        default_args=dict(
            model=model,
            optimizer=swa_optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    swa_runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        swa_optimizer_config = Fp16OptimizerHook(
            **cfg.swa_optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.swa_optimizer_config:
        swa_optimizer_config = OptimizerHook(**cfg.swa_optimizer_config)
    else:
        swa_optimizer_config = cfg.swa_optimizer_config

    # register hooks
    swa_runner.register_training_hooks(cfg.swa_lr_config, swa_optimizer_config,
                                       cfg.swa_checkpoint_config,
                                       cfg.log_config,
                                       cfg.get('momentum_config', None))
    if distributed:
        if isinstance(swa_runner, EpochBasedRunner):
            swa_runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHookMed if distributed else EvalHookMed
        # swa_runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
        swa_eval = True
        swa_eval_hook = eval_hook(val_dataloader,  **eval_cfg) #save_best='bbox_mAP',
    else:
        swa_eval = False
        swa_eval_hook = None

    # register swa hook
    swa_hook = SWAHook(
        swa_eval=swa_eval,
        eval_hook=swa_eval_hook,
        swa_interval=cfg.swa_interval)
    swa_runner.register_hook(swa_hook, priority='LOW') 

    # register user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            swa_runner.register_hook(hook, priority=priority)

    if cfg.swa_resume_from:
        swa_runner.resume(cfg.swa_resume_from)
    elif cfg.swa_load_from:
        # # use the best pretrained model as the starting model for swa training
        # if cfg.swa_load_from == 'best_bbox_mAP.pth':
        #     best_model_path = os.path.join(cfg.work_dir, cfg.swa_load_from)
        #     # avoid the best pretrained model being overwritten
        #     new_best_model_path = os.path.join(cfg.work_dir,
        #                                        'best_bbox_mAP_pretrained.pth')
        #     if swa_runner.rank == 0:
        #         import shutil
        #         assert os.path.exists(best_model_path)
        #         shutil.copy(
        #             best_model_path,
        #             new_best_model_path,
        #             follow_symlinks=False)
        #     cfg.swa_load_from = best_model_path
        swa_runner.load_checkpoint(cfg.swa_load_from)

    swa_runner.run(data_loaders, cfg.workflow)

    # TODO: caliberate bn
    # swa_update_bn(data_loaders[0], swa_runner.model)
    # if swa_runner.rank == 0: 
    #     swa_runner.save_checkpoint(swa_runner.work_dir, save_optimizer=False, 
    #                                 filename_tmpl='swa_model_final_update_bn_{}.pth')

