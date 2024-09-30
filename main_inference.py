import argparse
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from datasets.build_dataset import build_finetune_dataset
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from models import models_vit
from utils.loss import DiscreteLoss
from pathlib import Path
from utils.metrics import get_class_num_by_dataset
import timm.optim.optim_factory as optim_factory
from torchvision.utils import save_image as tv_save_image
def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification in multi-class task', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=-1,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=-1,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    # Dataset parameters
    parser.add_argument('--data_path', default='/data/cchuang/emoset/', type=str,
                        help='dataset path')

    parser.add_argument('--log_dir', default='/output/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='/data/cchuang/emoset/75_train_emo8/checkpoint-max.pth',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.set_defaults(eval=True)
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.set_defaults(dist_eval=True)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--output_dir', default='/data/cchuang/emoset/unix_infer/emo8', type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--dataset', default="emo8",
                        choices=["Emotion6", "FI", "UBE", "CAER-S", "EMOTIC", "HECO", "emo8"], type=str,
                        help='datasets.')
    parser.add_argument('--split_ratio', default=1, type=float, help='split ratio')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # simple augmentation
    dataset_test = build_finetune_dataset(False, args)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    nb_classes = get_class_num_by_dataset(args.dataset)
    model = models_vit.__dict__[args.model](
        num_classes=nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = DiscreteLoss('dynamic', device)
    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if args.eval:
        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank,
                shuffle=False) # shuffle=True to reduce monitor bias
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )

        # Perform inference and save misclassified samples
        perform_inferences(data_loader_test, model, device, args.output_dir)
        exit(0)


def perform_inferences(data_loader, model, device, output_dir):
    model.eval()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    all_samples_with_paths = data_loader.dataset.samples_with_paths

    with torch.no_grad():
        for idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Calculate the start index for the current batch
            batch_start_idx = idx * data_loader.batch_size

            for i in range(images.size(0)):
                sample_idx = batch_start_idx + i
                path, _ = all_samples_with_paths[sample_idx]
                original_filename = os.path.basename(path)
                pred_label = preds[i].item()
                true_label = labels[i].item()

                # Construct the new filename
                # new_filename = f"{original_filename.split('.')[0]}_{pred_label}_{true_label}.jpg"
                parent_folder_name = os.path.basename(os.path.dirname(path))
                new_filename = f"{parent_folder_name}_{original_filename.split('.')[0]}_{pred_label}_{true_label}.jpg"

                new_file_path = os.path.join(output_dir, new_filename)

                # Save the image
                image = images[i].cpu()
                save_tensor_image(image, new_file_path)
                print(f"Saved image to {new_file_path}")

    print(f"Finished saving images to {output_dir}")
    return

def perform_inference_all(data_loader, model, device, output_dir):
    model.eval()
    misclassified_samples = []
    correct_samples = []
    misclassified_true_labels = []
    correct_true_labels = []
    misclassified_pred_labels = []
    correct_pred_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Collect misclassified samples
            mask_misclassified = preds != labels
            mask_correct = preds == labels

            if mask_misclassified.any():
                misclassified_images = images[mask_misclassified]
                true_labels_misclassified = labels[mask_misclassified].cpu().numpy()
                pred_labels_misclassified = preds[mask_misclassified].cpu().numpy()

                for img, true_label, pred_label in zip(misclassified_images, true_labels_misclassified,
                                                       pred_labels_misclassified):
                    misclassified_samples.append(img.cpu())
                    misclassified_true_labels.append(true_label)
                    misclassified_pred_labels.append(pred_label)

            if mask_correct.any():
                correct_images = images[mask_correct]
                true_labels_correct = labels[mask_correct].cpu().numpy()
                pred_labels_correct = preds[mask_correct].cpu().numpy()

                for img, true_label, pred_label in zip(correct_images, true_labels_correct, pred_labels_correct):
                    correct_samples.append(img.cpu())
                    correct_true_labels.append(true_label)
                    correct_pred_labels.append(pred_label)

    # Save samples
    if output_dir:
        misclassified_dir = os.path.join(output_dir, 'misclassified')
        correct_dir = os.path.join(output_dir, 'correct')
        os.makedirs(misclassified_dir, exist_ok=True)
        os.makedirs(correct_dir, exist_ok=True)

        for i, (img, true_label, pred_label) in enumerate(
                zip(misclassified_samples, misclassified_true_labels, misclassified_pred_labels)):
            img_path = os.path.join(misclassified_dir, f"misclassified_{i}_true{true_label}_pred{pred_label}.png")
            save_tensor_image(img, img_path)
            print(f"Saved misclassified image {i} to {img_path}")

        for i, (img, true_label, pred_label) in enumerate(
                zip(correct_samples, correct_true_labels, correct_pred_labels)):
            img_path = os.path.join(correct_dir, f"correct_{i}_true{true_label}_pred{pred_label}.png")
            save_tensor_image(img, img_path)
            print(f"Saved correct image {i} to {img_path}")

    return misclassified_true_labels, misclassified_pred_labels, correct_true_labels, correct_pred_labels


def perform_inference(data_loader, model, device, output_dir):
    model.eval()
    misclassified_samples = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Collect misclassified samples
            mask = preds != labels
            if mask.any():
                misclassified_images = images[mask]
                misclassified_true_labels = labels[mask].cpu().numpy()
                misclassified_pred_labels = preds[mask].cpu().numpy()

                for img, true_label, pred_label in zip(misclassified_images, misclassified_true_labels,
                                                       misclassified_pred_labels):
                    misclassified_samples.append(img.cpu())
                    true_labels.append(true_label)
                    pred_labels.append(pred_label)

    # Save misclassified samples
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for i, (img, true_label, pred_label) in enumerate(zip(misclassified_samples, true_labels, pred_labels)):
            img_path = os.path.join(output_dir, f"misclassified_{i}_true{true_label}_pred{pred_label}.png")
            # Assuming `save_image` is a function to save images, replace it with your preferred method
            save_tensor_image(img, img_path)
            print(f"Saved misclassified image {i} to {img_path}")

    return true_labels, pred_labels

def save_tensor_image(tensor, path):
    # Assuming `tensor` is of shape (C, H, W) and values in [0, 1]
    tv_save_image(tensor, path)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)