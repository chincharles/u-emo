import math
import sys
from typing import Iterable
import torch
import utils.misc as misc
import utils.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, yolo_model: torch.nn.Module, clip_semactic: torch.nn.Module, teacher_clip: torch.nn.Module,
                    attn_fusion: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    components = [component for component in args.use_component]

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        image_context = samples['image'].to(device, non_blocking=True)
        image_body = samples['image_yolo'].to(device, non_blocking=True)
        image_body = yolo_model(image_body)
        text_features = clip_semactic(samples)
        with torch.cuda.amp.autocast():
            loss, _, _, cls_token1 = model(image_context, mask_ratio=args.mask_ratio)
            _, _, _, cls_token2 = model(image_body, mask_ratio=args.mask_ratio)


            fusion_token = attn_fusion(cls_token1, cls_token2)
            if 'fusion' in components:
                similarity_loss, contrastive_similarity = teacher_clip(image_context, fusion_token, text_features)
            else:
                similarity_loss, contrastive_similarity = teacher_clip(image_context, cls_token1, text_features)
        loss_value = configure_loss(components, loss.item(), similarity_loss.item(), contrastive_similarity.item())
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter
        loss_scaler(configure_loss(components, loss, similarity_loss, contrastive_similarity), optimizer,
                    parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        if 'mae_loss' in components:
            metric_logger.update(maeloss=loss.item())
        if 'similarity_loss' in components:
            metric_logger.update(similarity_loss=similarity_loss.item())
        if 'contrastive_similarity' in components:
            metric_logger.update(contrastive_similarity=contrastive_similarity.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)

            if 'mae_loss' in components:
                log_writer.add_scalar('mae_loss', loss.item(), epoch_1000x)
            if 'similarity_loss' in components:
                log_writer.add_scalar('similarity_loss', similarity_loss.item(), epoch_1000x)
            if 'contrastive_similarity' in components:
                log_writer.add_scalar('contrastive_similarity', contrastive_similarity.item(), epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def configure_loss(components, mae_loss, similarity_loss, contrastive_similarity):


    if 'fusion' in components:
        if lists_equal(components, ['fusion', 'mae_loss']):
            loss_sc = mae_loss
        if lists_equal(components, ['fusion', 'similarity_loss']):
            loss_sc = similarity_loss
        if lists_equal(components, ['fusion', 'contrastive_similarity']):
            loss_sc = contrastive_similarity

        if lists_equal(components, ['fusion', 'similarity_loss', 'mae_loss']):
            loss_sc = mae_loss + similarity_loss
        if lists_equal(components, ['fusion', 'mae_loss', 'contrastive_similarity']):
            loss_sc = mae_loss + contrastive_similarity
        if lists_equal(components, ['fusion', 'similarity_loss', 'contrastive_similarity']):
            loss_sc = similarity_loss + contrastive_similarity
        if lists_equal(components, ['fusion', 'mae_loss', 'similarity_loss', 'contrastive_similarity']):
            loss_sc = mae_loss + similarity_loss + contrastive_similarity
        return loss_sc


    if lists_equal(components, ['mae_loss']):
        loss_sc = mae_loss
    if lists_equal(components, ['similarity_loss']):
        loss_sc = similarity_loss
    if lists_equal(components, ['contrastive_similarity']):
        loss_sc = contrastive_similarity

    if lists_equal(components, ['similarity_loss', 'mae_loss']):
        loss_sc = mae_loss + similarity_loss
    if lists_equal(components, ['mae_loss', 'contrastive_similarity']):
        loss_sc = mae_loss + contrastive_similarity
    if lists_equal(components, ['similarity_loss', 'contrastive_similarity']):
        loss_sc = similarity_loss + contrastive_similarity
    if lists_equal(components, ['mae_loss', 'similarity_loss', 'contrastive_similarity']):
        loss_sc = mae_loss + similarity_loss + contrastive_similarity

    return loss_sc


def lists_equal(list1, list2):
    if len(list1) != len(list2):
        return False
    sorted_list1 = sorted(list1)
    sorted_list2 = sorted(list2)
    for str1, str2 in zip(sorted_list1, sorted_list2):
        if str1 != str2:
            return False
    return True