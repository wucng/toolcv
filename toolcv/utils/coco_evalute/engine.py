import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from . import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, lr_scheduler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    num_trains = len(data_loader.dataset)
    total_loss = 0.0
    # """
    # lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    # """

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        total_loss += losses.item()

    return total_loss / num_trains


def train_one_epoch2(model, loss_func, optimizer, data_loader, device, epoch, print_freq, writer, batch_size,
                     lr_scheduler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    num_trains = len(data_loader.dataset)
    total_loss = 0.0
    # """
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    # """
    for idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # images = list(image.to(device) for image in images)
        images = torch.stack(images, 0).to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: v.to(device) for k, v in targ.items() if k not in ["path"]} for targ in targets]

        # loss_dict = model(images, targets)
        output = model(images)
        loss_dict = loss_func(output, targets, True)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 记录到TensorBoard
        writer.add_scalar('total_loss', losses.item(), epoch * num_trains // batch_size + idx)
        for key, loss in loss_dict.items():
            writer.add_scalar(key, loss.item(), epoch * num_trains // batch_size + idx)

        total_loss += losses.item()

    writer.add_scalar("total_loss_epoch", total_loss, epoch)

    return total_loss / num_trains


def train_one_epoch3(model, optimizer, data_loader, device, epoch, print_freq, writer, batch_size, lr_scheduler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    num_trains = len(data_loader.dataset)
    total_loss = 0.0
    # """
    # lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    # """
    for idx, (data, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # """
        # 按边长最大填充
        bs = len(data)
        channels = data[0].size(0)
        fh = 0;
        fw = 0
        for da in data:
            _fh, _fw = da.shape[-2:]
            fh = max(fh, _fh)
            fw = max(fw, _fw)
        # 填充成统一大小
        new_data = torch.zeros([bs, channels, fh, fw], dtype=torch.float32)
        for i, da in enumerate(data):
            _fh, _fw = da.shape[-2:]
            new_data[i, :, 0:_fh, 0:_fw] = da
        data = new_data
        # """

        data = data.to(device)
        target = [{k: v.to(device) for k, v in targ.items() if k not in ["path"]} for targ in target]

        model.zero_grad()
        _, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, proposal = model(
            data, target)

        loss_dict = {"rpn_loss_cls": rpn_loss_cls, "rpn_loss_bbox": rpn_loss_bbox,
                     "RCNN_loss_cls": RCNN_loss_cls, "RCNN_loss_bbox": RCNN_loss_bbox}

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 记录到TensorBoard
        writer.add_scalar('total_loss', losses.item(), epoch * num_trains // batch_size + idx)
        for key, loss in loss_dict.items():
            writer.add_scalar(key, loss.item(), epoch * num_trains // batch_size + idx)

        total_loss += losses.item()

    writer.add_scalar("total_loss_epoch", total_loss, epoch)

    return total_loss / num_trains


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


@torch.no_grad()
def evaluate_do(self, iou_threshold, conf_threshold, with_nms):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    self.model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(self.val_dataloader.dataset)
    iou_types = _get_iou_types(self.model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(self.val_dataloader, 100, header):
        torch.cuda.synchronize()
        model_time = time.time()
        boxes, scores, labels = self.evalute_step((image, targets), 0, iou_threshold, conf_threshold, None, None, None,
                                                  with_nms)

        outputs = [{'boxes': boxes.to(cpu_device), 'scores': scores.to(cpu_device), 'labels': labels.to(cpu_device)}]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


@torch.no_grad()
def evaluate_dov2(self, **kwargs):
    """
    def evaluate_do(self, iou_threshold, conf_threshold, with_nms):
    """
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    self.model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(self.val_dataloader.dataset)
    # iou_types = _get_iou_types(self.model)
    iou_types = ["bbox"]
    if 'iou_types' in kwargs:
        if kwargs['iou_types'] == 'masks': iou_types.append("segm")
        if kwargs['iou_types'] == 'keypoints': iou_types.append("keypoints")

    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(self.val_dataloader, 100, header):
        torch.cuda.synchronize()
        model_time = time.time()
        preds = self.evalute_step((image, targets), 0, **kwargs)
        boxes, scores, labels = preds['boxes'], preds['scores'], preds['labels']
        outputs = [{'boxes': boxes.to(cpu_device), 'scores': scores.to(cpu_device), 'labels': labels.to(cpu_device)}]
        if 'masks' in preds:
            masks = preds['masks'].to(cpu_device)
            outputs[0].update(dict(masks=masks))
        if 'keypoints' in preds:
            keypoints = preds['keypoints'].to(cpu_device)
            outputs[0].update(dict(keypoints=keypoints))

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


@torch.no_grad()
def evaluate2(model, loss_func, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = torch.stack(image, 0)
        image = image.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        new_targets = [{k: v.to(device) for k, v in targ.items() if k not in ["boxes", "labels", "path"]} for targ in
                       targets]
        # targets = [{k: v.to(device) for k, v in targ.items()} for targ in targets]

        torch.cuda.synchronize()
        model_time = time.time()

        output = model(image)
        outputs = loss_func(output, new_targets)

        # outputs = [{k: torch.stack(v).to(cpu_device) for k, v in t.items()} for t in outputs]
        outputs_1 = []
        for t in outputs:
            if t is None:
                outputs_1.append({"boxes": torch.as_tensor([[0, 0, 0, 0]], dtype=torch.float32),
                                  "labels": torch.as_tensor([0], dtype=torch.long),
                                  "scores": torch.as_tensor([0.])})
            else:
                _dict = {k: torch.stack(v).to(cpu_device) for k, v in t.items()}
                outputs_1.append(_dict)
        outputs = outputs_1
        del outputs_1

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


@torch.no_grad()
def evaluate3(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for data, target in metric_logger.log_every(data_loader, 100, header):
        # """
        # 按边长最大填充
        bs = len(data)
        channels = data[0].size(0)
        fh = 0;
        fw = 0
        for da in data:
            _fh, _fw = da.shape[-2:]
            fh = max(fh, _fh)
            fw = max(fw, _fw)
        # 填充成统一大小
        new_data = torch.zeros([bs, channels, fh, fw], dtype=torch.float32)
        for i, da in enumerate(data):
            _fh, _fw = da.shape[-2:]
            new_data[i, :, 0:_fh, 0:_fw] = da
        data = new_data
        # """

        data = data.to(device)
        new_target = [
            {k: v.to(device) for k, v in targ.items() if k not in ["path", "boxes", "labels"]} for targ
            in target]

        torch.cuda.synchronize()
        model_time = time.time()

        rois, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, outputs = model(data, new_target)

        # outputs = [{k: torch.stack(v).to(cpu_device) for k, v in t.items()} for t in outputs]
        outputs_1 = []
        for t in outputs:
            if t is None:
                outputs_1.append({"boxes": torch.as_tensor([[0, 0, 0, 0]], dtype=torch.float32),
                                  "labels": torch.as_tensor([0], dtype=torch.long),
                                  "scores": torch.as_tensor([0.])})
            else:
                _dict = {k: v.to(cpu_device) for k, v in t.items()}  # torch.stack(v)
                outputs_1.append(_dict)
        outputs = outputs_1
        del outputs_1

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(target, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
