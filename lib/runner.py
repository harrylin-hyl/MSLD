import pickle
import random
import logging

import cv2
import torch
import numpy as np
from tqdm import tqdm, trange


class Runner:
    def __init__(self, cfg, exp, device, resume=False, view=None, deterministic=False):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.logger = logging.getLogger(__name__)

        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self):
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg['epochs']
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()
        for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.train()
            pbar = tqdm(train_loader)
            for i, (images, labels, _) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(images, **self.cfg.get_train_parameters())
                loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scheduler step (iteration based)
                scheduler.step()

                # Log
                postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
                postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)
                postfix_dict['loss'] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)

            # Validate
            if (epoch + 1) % self.cfg['val_every'] == 0:
                self.eval(epoch, on_val=True)
        self.exp.train_end_callback()

    def eval(self, epoch, on_val=False, save_predictions=False):
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        state_dict = self.exp.get_epoch_model(epoch)
        # Unused training parameters in some models 
        if "n_att_layer.conv.weight" in state_dict:
            del state_dict["n_att_layer.conv.weight"]
        if "n_att_layer.conv.bias" in state_dict:
            del state_dict["n_att_layer.conv.bias"]
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        if on_val:
            dataloader = self.get_val_dataloader()
        else:
            dataloader = self.get_test_dataloader()
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
            for idx, (images, _, _) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                output = model(images, **test_parameters)
                prediction = model.decode(output, as_lanes=True)
                # prediction = model.decode(output, as_lanes=True, **test_parameters)
                predictions.extend(prediction)
                if self.view:
                    img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    # cv2.imshow('img', img)
                    # label,_,_ = dataloader.dataset.draw_annotation(idx)
                    # cv2.imshow('label', label)
                    # pred,_,_ = dataloader.dataset.draw_annotation(idx, pred=prediction[0])
                    # cv2.imshow('pred_an', pred)
                    img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0])
                    if self.view == 'mistakes' and fp == 0 and fn == 0:
                        continue
                    cv2.imshow('pred', img)
                    cv2.waitKey(0)

        if save_predictions:
            with open('predictions.pkl', 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)

    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset('train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.cfg['batch_size'],
                                                   shuffle=True,
                                                   num_workers=8,
                                                   worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=8,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)

def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)