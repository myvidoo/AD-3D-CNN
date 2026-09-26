# -*- coding: utf-8 -*-
"""
5-fold cross-validation + equal-weight ensemble training/evaluation entry point.

Corresponding paper sections §2.4 / §3.2 (Table 2, confusion matrix, single-fold and
ensemble metrics).

Usage:
    python train/train_5fold.py

Notes:
    - Locks in the hyperparameters of the grid-search best Run 128 (read from
      all_training_results.csv);
    - early stopping monitors AUC only; checkpoints are saved based on the geometric mean
      GM=sqrt(AUC*F1);
    - the ensemble strategy is equal-weight averaging of logits.
"""

import os
import sys
import gc
import csv
import math
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config, resolve_path

from common.utils import (
    NUM_CLASSES, CLASS_NAMES, LABEL_MAP_FLOAT_TO_INT, NUM_WORKERS,
    EARLY_STOP_PATIENCE, GLOBAL_SEED,
    get_data_transforms, train_epoch, validate_epoch,
    safe_torch_load, load_dataset_from_excel, parse_config_from_string,
)
from model.densenet169_attention import get_model, get_loss_function, get_scheduler


def load_best_config_from_csv(run_id, grid_results_csv):
    """Extract the configuration for the specified run_id from the grid-search CSV"""
    if not os.path.exists(grid_results_csv):
        raise FileNotFoundError(f"global results file not found: {grid_results_csv}")
    df = pd.read_csv(grid_results_csv)
    row = df[df['run_id'] == run_id]
    if row.empty:
        raise ValueError(f"no record found for run_id = {run_id}")
    config = parse_config_from_string(row.iloc[0]['config'])
    print(f"[INFO] locked configuration: Run {run_id}")
    return config


def load_data_excluding_test(cfg):
    """Read the data, excluding the fixed test set"""
    data_root = cfg["data"].get("data_root") or None
    adni_excel = cfg["data"]["adni_excel"]
    fixed_split_path = resolve_path(cfg, "data.fixed_split")

    all_data = load_dataset_from_excel(adni_excel, data_root=data_root)

    if not os.path.exists(fixed_split_path):
        raise FileNotFoundError(f"test-set file not found: {fixed_split_path}")
    with open(fixed_split_path, 'r') as f:
        fixed_split = json.load(f)

    test_paths = {item['file_path'] for item in fixed_split.get('test_split', [])}
    test_data = [item for item in all_data if item['image'] in test_paths]
    cv_data = [item for item in all_data if item['image'] not in test_paths]
    cv_labels = [item['label'] for item in cv_data]

    print(f"[INFO] total data: {len(all_data)}, test set: {len(test_data)}, CV set: {len(cv_data)}")
    return cv_data, cv_labels, test_data


def train_single_fold(fold_idx, train_data, val_data, config, device, cv_results_dir, checkpoint_dir, num_epochs):
    print(f"\n{'=' * 40}\nStarting training for fold {fold_idx + 1}...\n{'=' * 40}")

    fold_dir = os.path.join(cv_results_dir, f"fold_{fold_idx + 1}")
    os.makedirs(fold_dir, exist_ok=True)
    epoch_csv = os.path.join(fold_dir, "epoch_metrics.csv")
    tensorboard_dir = os.path.join(cv_results_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, f"fold_{fold_idx + 1}"))

    a_max = config.get('a_max', 1100)
    train_transforms, val_transforms = get_data_transforms(
        augmentation_intensity=config['augmentation_intensity'], a_max=a_max
    )
    from monai.data import Dataset
    train_loader = DataLoader(Dataset(data=train_data, transform=train_transforms),
                              batch_size=config['batch_size'], shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(Dataset(data=val_data, transform=val_transforms),
                            batch_size=config['batch_size'], shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = get_model(config['model_name'], NUM_CLASSES, device,
                      config['dropout_rate'], config['attention_type'])
    class_counts = np.bincount([i['label'] for i in train_data], minlength=NUM_CLASSES)
    class_weights = torch.FloatTensor(
        (1.0 / (class_counts + 1e-6)) / (1.0 / (class_counts + 1e-6)).sum() * NUM_CLASSES
    ).to(device)

    loss_func = get_loss_function(config['loss_name'], weight=class_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = get_scheduler(config['scheduler_name'], optimizer, num_epochs, len(train_loader))

    # Resuming: detect the latest checkpoint and restore the training state
    best_composite = 0.0
    best_auc_early = 0.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold_idx + 1}_best_geo.pth")
    best_auc_record = 0.0
    best_f1_record = 0.0
    start_epoch = 0

    latest_path = os.path.join(checkpoint_dir, f"fold_{fold_idx + 1}_latest.pth")
    if os.path.exists(latest_path):
        ckpt = safe_torch_load(latest_path, map_location=device)
        model.base_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = int(ckpt.get('epoch', 0))
        extra = ckpt.get('extra_state', {}) or {}
        best_composite = extra.get('best_composite', 0.0)
        best_auc_early = extra.get('best_auc_early', 0.0)
        best_epoch = extra.get('best_epoch', 0)
        patience_counter = extra.get('patience_counter', 0)
        best_auc_record = extra.get('best_auc_record', 0.0)
        best_f1_record = extra.get('best_f1_record', 0.0)
        print(f"[Resume] Fold {fold_idx + 1} resuming training from epoch {start_epoch} "
              f"(GM={best_composite:.4f})")

    # When resuming, append to epoch_csv; on the first run, overwrite and write the header
    mode = 'a' if start_epoch > 0 else 'w'
    with open(epoch_csv, mode, newline='') as f:
        csv_writer = csv.writer(f)
        if start_epoch == 0:
            csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_auc', 'val_f1', 'learning_rate'])

        for epoch in range(start_epoch + 1, num_epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, loss_func, device,
                                     scheduler, config['grad_accum_steps'])
            val_metrics = validate_epoch(model, val_loader, loss_func, device, NUM_CLASSES,
                                         writer=writer, epoch=epoch)

            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['auc'])

            curr_auc = val_metrics['auc']
            curr_f1 = val_metrics['f1_macro']
            curr_lr = optimizer.param_groups[0]['lr']
            csv_writer.writerow([epoch, train_loss, val_metrics['loss'],
                                 val_metrics['accuracy'], curr_auc, curr_f1, curr_lr])

            # Saving criterion: geometric mean
            current_composite = math.sqrt(curr_auc * curr_f1)
            if current_composite > best_composite:
                best_composite = current_composite
                best_epoch = epoch
                best_auc_record = curr_auc
                best_f1_record = curr_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.base_model.state_dict() if hasattr(model, 'base_model') else model.state_dict(),
                    'composite_score': best_composite,
                    'metrics': val_metrics
                }, best_checkpoint_path)
                print(f"  [INFO] Epoch {epoch}: geometric mean improved to {best_composite:.4f} "
                      f"(AUC:{curr_auc:.4f}, F1:{curr_f1:.4f})")

            # Early stopping: based on AUC only
            if curr_auc > best_auc_early:
                best_auc_early = curr_auc
                patience_counter = 0
            else:
                patience_counter += 1

            # Save the overwriting "latest" checkpoint every epoch (with resumable state, loses at
            # most 1 epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.base_model.state_dict() if hasattr(model, 'base_model') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'extra_state': {
                    'best_composite': best_composite,
                    'best_auc_early': best_auc_early,
                    'best_epoch': best_epoch,
                    'patience_counter': patience_counter,
                    'best_auc_record': best_auc_record,
                    'best_f1_record': best_f1_record,
                },
            }, latest_path)

            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"[INFO] fold {fold_idx + 1} early-stopped at Epoch {epoch}")
                break

            if epoch % 20 == 0:
                print(f"  Fold {fold_idx + 1} Epoch {epoch} | GM:{current_composite:.4f} "
                      f"| AUC:{curr_auc:.4f} | F1:{curr_f1:.4f}")

    writer.close()
    print(f"[INFO] fold {fold_idx + 1} complete; best GM: {best_composite:.4f} (Epoch {best_epoch})")
    return {
        'fold': fold_idx + 1,
        'best_auc': best_auc_record,
        'best_f1': best_f1_record,
        'checkpoint': best_checkpoint_path
    }


def compute_detailed_metrics(y_true, y_pred, y_proba):
    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
    cm = confusion_matrix(y_true, y_pred)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    class_aucs = {CLASS_NAMES[i]: roc_auc_score((y_true == i).astype(int), y_proba[:, i])
                  for i in range(NUM_CLASSES)}
    class_report = {CLASS_NAMES[i]: {'precision': p[i], 'recall': r[i], 'f1': f[i],
                                     'auc': class_aucs[CLASS_NAMES[i]]}
                    for i in range(NUM_CLASSES)}
    return {'acc': acc, 'f1': f1, 'auc': auc, 'cm': cm, 'class_report': class_report}


def test_ensemble(checkpoint_paths, test_files, config, device):
    from monai.data import Dataset
    _, test_transforms = get_data_transforms(0, a_max=config.get('a_max', 1100))
    test_data = [{'image': f['image'], 'label': f['label']} for f in test_files
                 if os.path.exists(f['image'])]

    if len(test_data) != len(test_files):
        raise FileNotFoundError(f"{len(test_files) - len(test_data)} files are missing from the test set")

    test_loader = DataLoader(Dataset(test_data, transform=test_transforms), batch_size=1,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    all_logits = []
    single_results = []

    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = get_model(config['model_name'], NUM_CLASSES, device,
                          config['dropout_rate'], config['attention_type'])
        state_dict = checkpoint['model_state_dict']
        if hasattr(model, 'base_model') and not any(k.startswith('base_model.') for k in state_dict):
            model.base_model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

        model.eval()
        fold_y_true, fold_y_pred, fold_y_proba = [], [], []
        fold_logits = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)

                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                fold_y_true.extend(labels.cpu().numpy())
                fold_y_pred.extend(preds.cpu().numpy())
                fold_y_proba.extend(probs.cpu().numpy())
                fold_logits.extend(outputs.cpu().numpy())

        single_results.append(compute_detailed_metrics(fold_y_true, fold_y_pred, fold_y_proba))
        all_logits.append(np.array(fold_logits))

        del model
        gc.collect()

    weights = np.ones(5) / 5
    mean_logits = np.sum(np.stack(all_logits, axis=0) * weights[:, None, None], axis=0)
    ensemble_probs = F.softmax(torch.tensor(mean_logits, dtype=torch.float32), dim=1).numpy()
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    y_true = [item['label'] for item in test_data]

    ensemble_metrics = compute_detailed_metrics(y_true, ensemble_preds, ensemble_probs)
    return single_results, ensemble_metrics, weights


def save_detailed_results(single_results, ensemble_results, cv_results_dir):
    for i, result in enumerate(single_results):
        fold_name = f"fold_{i + 1}"
        pd.DataFrame(result['class_report']).T.to_csv(
            os.path.join(cv_results_dir, f'{fold_name}_class_metrics.csv'))
        pd.DataFrame(result['cm'], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            os.path.join(cv_results_dir, f'{fold_name}_confusion_matrix.csv'))

    pd.DataFrame(ensemble_results['class_report']).T.to_csv(
        os.path.join(cv_results_dir, 'ensemble_class_metrics.csv'))
    pd.DataFrame(ensemble_results['cm'], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        os.path.join(cv_results_dir, 'ensemble_confusion_matrix.csv'))


def main():
    parser = argparse.ArgumentParser(description="5-fold cross-validation + ensemble")
    parser.add_argument("--run_id", type=int, default=None, help="Grid-search Run ID (defaults to config best_run_id)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (defaults to num_epochs=200 from the Run configuration)")
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    run_id = args.run_id or cfg["best_run_id"]
    grid_results = resolve_path(cfg, "data.grid_results")
    config = load_best_config_from_csv(run_id, grid_results)

    output_dir = cfg["output_dir"]
    cv_results_dir = os.path.join(output_dir, "results_5fold_geo_resume")
    checkpoint_dir = os.path.join(cv_results_dir, "checkpoints")
    os.makedirs(cv_results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    cv_state_json = os.path.join(cv_results_dir, f"cv_state_run_{run_id}.json")

    cv_data, cv_labels, test_files = load_data_excluding_test(cfg)

    # State management (resumable training)
    state = None
    if os.path.exists(cv_state_json):
        with open(cv_state_json, 'r') as f:
            state = json.load(f)
        if state.get('current_fold', 0) >= 5:
            print("[INFO] Training already complete; proceeding directly to the test phase.")
        else:
            print(f"[INFO] Resuming from fold {state['current_fold'] + 1}...")
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
        state = {"run_id": run_id, "completed_folds": [], "current_fold": 0,
                 "fold_results": [], "fold_splits": []}
        for train_idx, val_idx in skf.split(np.zeros(len(cv_labels)), cv_labels):
            state["fold_splits"].append({"train_indices": train_idx.tolist(),
                                         "val_indices": val_idx.tolist()})
        with open(cv_state_json, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"[INFO] state saved: {cv_state_json}")

    num_epochs = args.epochs if args.epochs is not None else config.get('num_epochs', 200)

    if state.get('current_fold', 0) < 5:
        while state['current_fold'] < 5:
            fold_idx = state['current_fold']
            split_info = state['fold_splits'][fold_idx]
            train_data = [cv_data[i] for i in split_info['train_indices']]
            val_data = [cv_data[i] for i in split_info['val_indices']]

            result = train_single_fold(fold_idx, train_data, val_data, config, device,
                                       cv_results_dir, checkpoint_dir, num_epochs)

            state['completed_folds'].append(fold_idx)
            state['fold_results'].append(result)
            state['current_fold'] = fold_idx + 1
            with open(cv_state_json, 'w') as f:
                json.dump(state, f, indent=2)

            gc.collect()
            torch.cuda.empty_cache()

    checkpoint_paths = [r['checkpoint'] for r in state['fold_results']]
    if len(checkpoint_paths) != 5:
        raise ValueError(f"could not find the complete 5-fold checkpoints; current count: {len(checkpoint_paths)}")

    single_results, ensemble_results, weights = test_ensemble(checkpoint_paths, test_files, config, device)

    print(f"\n{'=' * 60}\n[INFO] Final test-set report\n{'=' * 60}")
    for i, r in enumerate(single_results):
        print(f"Fold {i + 1}: Acc={r['acc']:.4f}, F1={r['f1']:.4f}, AUC={r['auc']:.4f}")
    print(f"Ensemble: Acc={ensemble_results['acc']:.4f}, F1={ensemble_results['f1']:.4f}, "
          f"AUC={ensemble_results['auc']:.4f}")

    save_detailed_results(single_results, ensemble_results, cv_results_dir)
    pd.DataFrame([{'Fold': i + 1, 'Acc': r['acc'], 'F1': r['f1'], 'AUC': r['auc']}
                  for i, r in enumerate(single_results)]).to_csv(
        os.path.join(cv_results_dir, 'single_fold_test_results.csv'), index=False)
    pd.DataFrame([{'type': 'Ensemble', 'Acc': ensemble_results['acc'],
                   'F1': ensemble_results['f1'], 'AUC': ensemble_results['auc']}]).to_csv(
        os.path.join(cv_results_dir, 'ensemble_test_result.csv'), index=False)

    print(f"\n[INFO] training and inference complete. Results saved in: {cv_results_dir}")


if __name__ == "__main__":
    main()
