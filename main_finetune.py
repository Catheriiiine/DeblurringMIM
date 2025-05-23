# Copyright (c) 2022 Alpha-VL
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE:  https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import timm

#assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.misc as misc
from util.datasets import build_dataset, build_dataset_test
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from model import models_vit, models_convvit, models_effi

from engine_finetune import train_one_epoch, evaluate, evaluate_test
from engine_finetune import evaluate_bootstrap


import copy
#fix
# import pdb; pdb.set_trace()


def get_args_parser():
    parser = argparse.ArgumentParser('ConvMAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    # vit_base_patch16, vit_large_patch16, vit_huge_patch14 
    # convvit_base_patch16, convvit_large_patch16, convvit_huge_patch16
    parser.add_argument('--model', default='convvit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--early_stops', default=200, type=int,
                        help='epochs for early stopping')
    
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
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
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

    return parser


def main(args):

    misc.init_distributed_mode(args)

    # print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)
    dataset_test = build_dataset_test(is_train=False, args=args)

    # #fix: dummy dataset
    # # Define parameters for dummy data
    # batch_size = 4  # Set the batch size
    # num_classes = 2  # Adjust based on your model's output classes
    # image_size = (3, 224, 224)  # Typical image size for CNN models

    # # Generate random dummy images and labels
    # dummy_train_images = torch.randint(0, 256, (batch_size, *image_size), dtype=torch.uint8)
    # dummy_train_labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

    # dummy_val_images = torch.randint(0, 256, (batch_size, *image_size), dtype=torch.uint8)
    # dummy_val_labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

    # dummy_test_images = torch.randint(0, 256, (batch_size, *image_size), dtype=torch.uint8)
    # dummy_test_labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
    # dummy_train_images = dummy_train_images.float().div(255).half().to(device)
    # dummy_val_images = dummy_val_images.float().div(255).half().to(device)
    # dummy_test_images = dummy_test_images.float().div(255).half().to(device)



    # # Convert to TensorDataset
    # dataset_train = TensorDataset(dummy_train_images, dummy_train_labels)
    # dataset_val = TensorDataset(dummy_val_images, dummy_val_labels)
    # dataset_test = TensorDataset(dummy_test_images, dummy_test_labels)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        #print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.output_dir is not None and not args.eval:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
       # print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    # Convvit
    if "convvit_base_patch16" == args.model or "convvit_large_patch16" == args.model or "convvit_huge_patch16" == args.model:
        model = models_convvit.__dict__[args.model](
            drop_rate=0.0,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    # vit
    elif "vit_base_patch16" == args.model or "vit_large_patch16" == args.model or "vit_huge_patch14" == args.model:
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    
    elif "efficientnet_from_knees"  == args.model:
        model = models_effi.__dict__[args.model](out_features=1)
    else:
        model = timm.create_model(args.model,
                               pretrained=True,
                               num_classes=args.nb_classes,
                               drop_path_rate=args.drop_path)

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
#        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            # fix:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # assert set(msg.missing_keys) == {'classifier.1.weight', 'classifier.1.bias'}
            pass

        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        # fix
        trunc_normal_(model.head.weight, std=2e-5)
        # trunc_normal_(model.model.classifier[1].weight, std=2e-5)


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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    # param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
    #     no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #     layer_decay=args.layer_decay
    # )
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
        print("1")
    elif args.smoothing > 0.:
        
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        print(f"Label Smoothing Value: {args.smoothing}")
        print("2")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print("3")

    #print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    model_best = copy.deepcopy(model)

    if args.eval:
        correct_paths_pos = []    # Correctly classified and label==positive (1)
        correct_paths_neg = []    # Correctly classified and label==negative (0)
        incorrect_paths_pos = []  # Incorrectly classified and label==positive (1)
        incorrect_paths_neg = []
        test_stats = evaluate(data_loader_test, model_best, device)
        print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
        acc, f1, precision, recall, cls_report, auroc = evaluate_test(dataset_test, model_best, device)
        print ("Acc:", acc*100)
        print ("F1:", f1*100)
        print (cls_report)
        
        with open(os.path.join(args.output_dir, "res.txt"), mode="w+", encoding="utf-8") as f:
            f.writelines("---"*10+"\n") 
            f.writelines("Acc:" + str(acc*100) + "\n")
            f.writelines("F1:" + str( f1*100) + "\n")
            f.writelines("precision:" + str(precision*100) + "\n")
            f.writelines("recall:" + str(recall*100) + "\n")
            f.writelines("auroc:" + str(auroc*100) + "\n")
            f.writelines(cls_report + "\n")
            f.writelines("---"*10) 
        mean_diff, bootstrap_differences = evaluate_bootstrap(dataset_test, model_best, device, n_bootstrap=5000)
        print(f"Bootstrap mean difference (over 5000 iterations): {mean_diff*100:.2f}%")
        
        all_paths = [path for path, _ in dataset_test.samples]
        global_index = 0
        model_best.eval()
        with torch.no_grad():
            for images, labels in data_loader_test:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model_best(images)
                preds = outputs.argmax(dim=1)
                
                batch_size = images.size(0)
                for i in range(batch_size):
                    # Retrieve file path by matching the global sample index.
                    path = all_paths[global_index]
                    true_label = labels[i].item()    # 0 for negative, 1 for positive
                    pred_label = preds[i].item()
                    
                    if pred_label == true_label:  # Correct classification
                        if true_label == 1 and len(correct_paths_pos) < 5:
                            correct_paths_pos.append(path)
                        elif true_label == 0 and len(correct_paths_neg) < 5:
                            correct_paths_neg.append(path)
                    else:  # Incorrect classification
                        if true_label == 1 and len(incorrect_paths_pos) < 5:
                            incorrect_paths_pos.append(path)
                        elif true_label == 0 and len(incorrect_paths_neg) < 5:
                            incorrect_paths_neg.append(path)
                    
                    global_index += 1

                    # Check if we have collected 5 for each category.
                    if (len(correct_paths_pos) >= 5 and len(correct_paths_neg) >= 5 and
                        len(incorrect_paths_pos) >= 5 and len(incorrect_paths_neg) >= 5):
                        break
                # If all lists have at least 5 items, break out of the outer loop.
                if (len(correct_paths_pos) >= 5 and len(correct_paths_neg) >= 5 and
                    len(incorrect_paths_pos) >= 5 and len(incorrect_paths_neg) >= 5):
                    break
        
        # Print collected paths:
        print("\nPaths to 5 correctly classified positive images:")
        for p in correct_paths_pos[:5]:
            print(p)

        print("\nPaths to 5 correctly classified negative images:")
        for p in correct_paths_neg[:5]:
            print(p)

        print("\nPaths to 5 incorrectly classified positive images:")
        for p in incorrect_paths_pos[:5]:
            print(p)

        print("\nPaths to 5 incorrectly classified negative images:")
        for p in incorrect_paths_neg[:5]:
            print(p)
            
        exit(0)

        
        
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_epoch = -1
    for epoch in range(args.start_epoch, args.epochs):
        if (epoch - max_epoch > args.early_stops and max_epoch != -1):
            test_stats = evaluate(data_loader_test, model_best, device)
            print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
            acc, f1, precision, recall, cls_report, auroc = evaluate_test(dataset_test, model_best, device)
            print ("Acc:", acc*100)
            print ("F1:", f1*100)
            print (cls_report)
            
            with open(os.path.join(args.output_dir, "result.txt"), mode="a+", encoding="utf-8") as f:
                f.writelines("---"*10+"\n")
                f.writelines("Acc:" + str(acc*100) + "\n")
                f.writelines("F1:" + str( f1*100) + "\n")
                f.writelines("precision:" + str(precision*100) + "\n")
                f.writelines("recall:" + str(recall*100) + "\n")
                f.writelines("auroc:" + str(auroc*100) + "\n")
                f.writelines(cls_report + "\n")
                f.writelines("---"*10)
            exit(0)
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        if args.output_dir:
            if test_stats["acc1"] > max_accuracy:
                # if max_epoch != -1:
                #     old_path = args.output_dir + "/checkpoint-" + str(max_epoch) + ".pth"
                #     os.remove(old_path)
                max_epoch = epoch
                model_best = copy.deepcopy(model)
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'max_epoch': max_epoch }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




if __name__ == '__main__':
    # print(torch.version.cuda)     # e.g. '11.7' or '12.1'
    # print(torch.version.__version__)  # shows PyTorch version
    # print(torch.cuda.is_available())
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

