# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score, roc_curve, auc, confusion_matrix
import numpy as np


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            print("done")

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            # print("Input min:", samples.min().item(), "Input max:", samples.max().item())
            # print("Target min:", targets.min().item(), "Target max:", targets.max().item())
            loss = criterion(outputs, targets)
            #fix
            # Before backward pass
            # print("Loss before backward:", loss.item())  # Will raise error if NaN
            # assert torch.isfinite(loss).all(), "NaN/Inf detected in loss"
            # print("loss value:", loss.item())
            # assert torch.isfinite(samples).all(), "NaN/Inf detected in inputs"
            # assert torch.isfinite(targets).all(), "NaN/Inf detected in labels"
            # print("Input min:", samples.min().item(), "Input max:", samples.max().item())
            # print("Target min:", targets.min().item(), "Target max:", targets.max().item())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        # print("debug loss", loss.item())
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        #fix: train_acc
        batch_acc1, _ = accuracy(outputs, targets, topk=(1, 1))
        metric_logger.update(train_acc=batch_acc1.item())

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    print("Averaged TRAIN stats:", {k: meter.global_avg for k, meter in metric_logger.meters.items()})

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




from sklearn.metrics import f1_score, jaccard_score, accuracy_score,\
    classification_report, precision_score, recall_score, roc_curve, auc
@torch.no_grad()
def evaluate_test(dataset, model, device):
    all_label_gt = []
    all_label_pred = []
    all_pred = []
    model.eval()
    for i in range(len(dataset)):
        image, target = dataset[i]
        x_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pr_label = torch.sigmoid(model(x_tensor))
            _, preds = torch.max(pr_label, 1)
            pred_label = preds.cpu().numpy()[0]
        all_label_pred.append(pred_label)
        all_label_gt.append(target)
        all_pred.append(pr_label[0, 1].cpu().numpy())
    cls_report = classification_report(all_label_gt, all_label_pred, digits=4)
    print(cls_report)
    acc = accuracy_score(all_label_gt, all_label_pred)
    f1_micro = f1_score(all_label_gt, all_label_pred, average='micro')
    precision = precision_score(all_label_gt, all_label_pred, average='binary')
    recall = recall_score(all_label_gt, all_label_pred, average='binary')
    
    # Compute confusion matrix and then specificity
    cm = confusion_matrix(all_label_gt, all_label_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)  # Same as recall

    fpr, tpr, thresholds = roc_curve(all_label_gt, all_pred, pos_label = 1)
    auroc = auc(fpr, tpr)
    print("Accuracy:", acc * 100)
    print("F1 (micro):", f1_micro * 100)
    print("Precision:", precision * 100)
    print("Recall (Sensitivity):", sensitivity * 100)
    print("Specificity:", specificity * 100)
    print("AUROC:", auroc * 100)
    return acc, f1_micro, precision, recall, cls_report, auroc




@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    all_label_gt = []
    all_label_pred = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        all_label_gt.extend(target.numpy())
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
            _, output_1 = torch.max(torch.sigmoid(output), 1)
            output_1 = output_1.cpu().numpy()


        all_label_pred.extend(output_1)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
      #  metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    f1_weighted = f1_score(all_label_gt, all_label_pred, average='weighted') * 100
    print ("f1_weighted:", f1_weighted)
    metric_logger.meters['f1'].update(f1_weighted)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# @torch.no_grad()
# def evaluate_bootstrap(dataset, model, device, n_bootstrap=5000):
#     """
#     Evaluate the test set once per image, store the correctness result,
#     and perform bootstrapping to estimate the accuracy distribution.
    
#     Parameters:
#         dataset: The test dataset.
#         model: The trained model.
#         device: The device to run evaluation on.
#         n_bootstrap: Number of bootstrap iterations (default: 5000).
    
#     Returns:
#         mean_bootstrap_acc: Mean accuracy computed from bootstrap samples.
#         bootstrap_accuracies: A NumPy array of accuracy values from each bootstrap iteration.
#     """
#     model.eval()
#     all_correct = []

#     # Evaluate each sample only once.
#     for i in range(len(dataset)):
#         image, target = dataset[i]
#         x_tensor = image.unsqueeze(0).to(device)
        
#         # Convert target to tensor if it's not already one.
#         if not isinstance(target, torch.Tensor):
#             target = torch.tensor(target, device=device)
#         else:
#             target = target.to(device)
        
#         output = model(x_tensor)
#         # For binary classification with 2 outputs, use sigmoid then argmax.
#         pr_label = torch.sigmoid(output)
#         _, pred = torch.max(pr_label, 1)
#         correct = 1 if pred.item() == target.item() else 0
#         all_correct.append(correct)
    
#     all_correct = np.array(all_correct)
#     n_samples = len(dataset)
#     bootstrap_accuracies = []

#     # Optionally set a seed for reproducibility
#     # np.random.seed(42)
    
#     # Bootstrapping: for each iteration, sample with replacement
#     for _ in range(n_bootstrap):
#         indices = np.random.choice(n_samples, size=n_samples, replace=True)
#         bootstrap_acc = all_correct[indices].mean()
#         bootstrap_accuracies.append(bootstrap_acc)
    
#     bootstrap_accuracies = np.array(bootstrap_accuracies)
#     mean_bootstrap_acc = bootstrap_accuracies.mean()
#     return mean_bootstrap_acc, bootstrap_accuracies


@torch.no_grad()
def evaluate_bootstrap(dataset, model, device, 
                       baseline_pred_file='/home/catherine/Desktop/Thickened-Synovium/baseline_predictions.npz', 
                       n_bootstrap=5000):
    """
    Evaluate the primary model on the test set and load baseline predictions from a saved file.
    Then perform bootstrapping to estimate the distribution of the difference in accuracies
    (primary model accuracy - baseline model accuracy).

    Assumes that the test dataset and the baseline file have the same number of samples.

    Parameters:
        dataset: The test dataset.
        model: The primary (trained) model.
        device: The device to run evaluation on.
        baseline_pred_file: Path to the saved baseline predictions file.
        n_bootstrap: Number of bootstrap iterations (default: 5000).

    Returns:
        mean_diff: The mean difference (model accuracy - baseline accuracy) computed from bootstrap samples.
        bootstrap_differences: A NumPy array of differences from each bootstrap iteration.
    """
    model.eval()
    all_correct_model = []

    # Evaluate primary model predictions on each sample.
    for i in range(len(dataset)):
        image, target = dataset[i]
        x_tensor = image.unsqueeze(0).to(device)
        
        # Convert target to an integer.
        if not isinstance(target, torch.Tensor):
            target_val = int(target)
        else:
            target_val = target.item() if target.dim() == 0 else target[0].item()
        
        output = model(x_tensor)
        pr_model = torch.sigmoid(output)
        _, pred_model = torch.max(pr_model, 1)
        correct_model = 1 if pred_model.item() == target_val else 0
        all_correct_model.append(correct_model)
    
    all_correct_model = np.array(all_correct_model)
    n_samples = len(dataset)
    print("Number of samples in dataset:", n_samples)

    # Load baseline predictions (assumed saved as 'preds' and 'targets').
    baseline_data = np.load(baseline_pred_file)
    baseline_preds = baseline_data['preds']
    baseline_targets = baseline_data['targets']
    baseline_preds = (baseline_preds >= 0.5).astype(np.int64)
    print(baseline_preds)
    print("Length of baseline predictions:", len(baseline_preds))
    
    # Compute baseline correctness for each sample.
    all_correct_baseline = np.array([
        1 if baseline_preds[i] == baseline_targets[i] else 0
        for i in range(n_samples)
    ])

    # Print overall accuracies for verification.
    overall_model_acc = all_correct_model.mean()
    overall_baseline_acc = all_correct_baseline.mean()
    print("Overall primary model accuracy: {:.2%}".format(overall_model_acc))
    print("Overall baseline accuracy: {:.2%}".format(overall_baseline_acc))
    print("Overall difference: {:.2%}".format(overall_model_acc - overall_baseline_acc))

    bootstrap_differences = []
    np.random.seed(42)  # For reproducibility.
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        acc_model = all_correct_model[indices].mean()
        acc_baseline = all_correct_baseline[indices].mean()
        diff = acc_model - acc_baseline
        bootstrap_differences.append(diff)
    
    bootstrap_differences = np.array(bootstrap_differences)
    mean_diff = bootstrap_differences.mean()
    
    alpha = 0.05
    # Calculate CI
    lower = np.percentile(bootstrap_differences, 100 * (alpha / 2))
    upper = np.percentile(bootstrap_differences, 100 * (1 - alpha / 2))
    ci = (lower, upper)

    print(f"Bootstrap mean difference: {mean_diff * 100:.2f}%")
    print(f"{int((1 - alpha) * 100)}% Confidence Interval: ({lower * 100:.2f}%, {upper * 100:.2f}%)")

    return mean_diff, bootstrap_differences

# @torch.no_grad()
# def evaluate_bootstrap(dataset, model, device, baseline_pred_file='/home/catherine/Desktop/Thickened-Synovium/baseline_predictions.npz', n_bootstrap=5000):
    """
    Evaluate the primary model on the test set and load baseline predictions from a saved file.
    Then perform bootstrapping to estimate the distribution of the difference in accuracies
    (primary model accuracy - baseline model accuracy).

    Parameters:
        dataset: The test dataset.
        model: The primary (trained) model.
        baseline_pred_file: Path to the saved baseline predictions file (e.g., "baseline_predictions.npz").
        device: The device to run evaluation on.
        n_bootstrap: Number of bootstrap iterations (default: 5000).

    Returns:
        mean_diff: The mean difference (model accuracy - baseline accuracy) computed from bootstrap samples.
        bootstrap_differences: A NumPy array of the difference from each bootstrap iteration.
    """
    model.eval()
    all_correct_model = []

    # Evaluate primary model once on each test sample.
    for i in range(len(dataset)):
        image, target = dataset[i]
        x_tensor = image.unsqueeze(0).to(device)
        
        # Convert target to an integer.
        if not isinstance(target, torch.Tensor):
            target_val = int(target)
        else:
            target_val = target.item() if target.dim() == 0 else target[0].item()
        
        output = model(x_tensor)
        pr_model = torch.sigmoid(output)
        _, pred_model = torch.max(pr_model, 1)
        correct_model = 1 if pred_model.item() == target_val else 0
        all_correct_model.append(correct_model)
    
    all_correct_model = np.array(all_correct_model)
    n_samples = len(dataset)
    print(n_samples)

    # Load baseline predictions (assumed to have been saved as 'preds' and 'targets').
    baseline_data = np.load(baseline_pred_file)
    baseline_preds = baseline_data['preds']
    baseline_targets = baseline_data['targets']
    print(len(baseline_preds))
    print(len(baseline_targets))
    
    # Compute baseline correctness for each sample.
    # (Assuming baseline_preds and baseline_targets are aligned with the dataset order.)
    all_correct_baseline = np.array([1 if baseline_preds[i] == baseline_targets[i] else 0 for i in range(len(baseline_targets))])
    
    bootstrap_differences = []
    np.random.seed(42)  # For reproducibility.

    # Perform bootstrapping.
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        acc_model = all_correct_model[indices].mean()
        acc_baseline = all_correct_baseline[indices].mean()
        print(acc_model)
        print(acc_baseline)
        diff = acc_model - acc_baseline
        bootstrap_differences.append(diff)
    
    bootstrap_differences = np.array(bootstrap_differences)
    mean_diff = bootstrap_differences.mean()
    return mean_diff, bootstrap_differences