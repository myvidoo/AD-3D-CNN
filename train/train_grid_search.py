# -*- coding: utf-8 -*-
"""
Grid-search training entry point: 3D DenseNet-169 + switchable attention mechanisms,
three-class MRI.

Corresponding paper sections §2.3 / §3.1 (256-run hyperparameter grid search, best Run 128).

Usage:
    python train/train_grid_search.py [--max_runs N] [--resume_from M]

All paths are read from config.yaml; there are no hard-coded absolute paths.
"""

import os
import sys
import ast
import itertools
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config, resolve_path

from common.utils import (
    NUM_CLASSES, LABEL_MAP_FLOAT_TO_INT, NUM_WORKERS, CHECKPOINT_INTERVAL,
    EARLY_STOP_PATIENCE, GLOBAL_SEED,
    get_data_transforms, worker_init_fn, train_epoch, validate_epoch,
    save_checkpoint, safe_torch_load, find_latest_checkpoint, append_global_training_result,
    save_fixed_data_split, load_fixed_data_split, load_dataset_from_excel,
)
from model.densenet169_attention import get_model, get_loss_function, get_scheduler


# ==================== Hyperparameter grid definition ====================#

SCHEDULER_OPTIONS = ["reduce_on_plateau"]
LOSS_OPTIONS = ["ce"]
MODEL_OPTIONS = ["densenet169"]
ATTENTION_OPTIONS = ["none", "cbam", "eca", "axial"]
DROPOUT_OPTIONS = [0.0, 0.2]
GRAD_ACCUM_OPTIONS = [1, 2]
AUGMENTATION_OPTIONS = [0, 1]
BATCH_SIZE_OPTIONS = [3, 5]
LEARNING_RATE_OPTIONS = [1e-5, 2e-5, 5e-5, 1e-4]
EPOCH_OPTIONS = [200]
A_MAX_OPTIONS = [1100]
DEFAULT_A_MAX = A_MAX_OPTIONS[0]

DATA_SPLIT_OPTIONS = [{"train": 0.7, "val": 0.15, "test": 0.15}]
DATA_SPLIT = DATA_SPLIT_OPTIONS[0]


def generate_hyperparameter_combinations(epochs=None):
    """Generate all hyperparameter combinations (the 256 of paper §3.1). epochs defaults to EPOCH_OPTIONS[0]."""
    num_epochs = epochs if epochs is not None else EPOCH_OPTIONS[0]
    combinations = []

    base_params = itertools.product(
        MODEL_OPTIONS, SCHEDULER_OPTIONS, LOSS_OPTIONS, DROPOUT_OPTIONS,
        GRAD_ACCUM_OPTIONS, AUGMENTATION_OPTIONS, BATCH_SIZE_OPTIONS, LEARNING_RATE_OPTIONS,
    )

    for model, sched, loss, dropout, grad_acc, aug, bs, lr in base_params:
        attention_options = ATTENTION_OPTIONS if model == "densenet169" else ["none"]
        for attention_type in attention_options:
            split = DATA_SPLIT
            config = {
                'model_name': model,
                'scheduler_name': sched,
                'loss_name': loss,
                'attention_type': attention_type,
                'use_amp': False,
                'dropout_rate': dropout,
                'grad_accum_steps': grad_acc,
                'augmentation_intensity': aug,
                'batch_size': bs,
                'learning_rate': lr,
                'num_epochs': num_epochs,
                'train_ratio': split['train'],
                'val_ratio': split['val'],
                'test_ratio': split['test'],
                'a_max': DEFAULT_A_MAX
            }
            combinations.append(config)

    return combinations


def run_single_training(config, run_id, cfg):
    """Run a single training configuration"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = cfg["output_dir"]
    data_root = cfg["data"].get("data_root") or None
    adni_excel = cfg["data"].get("adni_excel")
    fixed_split_path = resolve_path(cfg, "data.fixed_split")
    global_csv = os.path.join(output_dir, "all_training_results.csv")

    print(f"\n{'=' * 80}\nRUN {run_id}: {config}\n{'=' * 80}")

    model_name = config['model_name']
    scheduler_name = config['scheduler_name']
    loss_name = config['loss_name']
    attention_type = config.get('attention_type', 'none')
    use_attention = attention_type != 'none'
    dropout_rate = config['dropout_rate']
    grad_accum_steps = config['grad_accum_steps']
    augmentation_intensity = config['augmentation_intensity']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    train_ratio = config['train_ratio']
    val_ratio = config['val_ratio']
    test_ratio = config['test_ratio']
    a_max = config['a_max']

    # Create the run directory (relative to output_dir, no absolute paths)
    attn_tag = f"_Attn_{attention_type}" if use_attention and model_name == "densenet169" else ""
    run_dir = os.path.join(
        output_dir,
        f"Grid_SEARCH_{run_id}_{model_name}{attn_tag}_dropout_{dropout_rate}_aug_{augmentation_intensity}"
        f"_bs_{batch_size}_lr_{learning_rate:.0e}_amax_{a_max}_sched_{scheduler_name}_loss_{loss_name}"
    )
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = None
    try:
        writer = SummaryWriter(log_dir=log_dir)

        # 1. Load data
        data_list = load_dataset_from_excel(adni_excel, data_root=data_root)
        all_labels = [item["label"] for item in data_list]

        # 2. Dataset splitting (reuse the fixed test set where possible)
        fixed_split = load_fixed_data_split(fixed_split_path)
        if fixed_split and 'test_split' in fixed_split:
            # fixed_split holds relative file names/sub-paths with the path stripped, so data_root
            # must be joined back on to obtain absolute paths
            fixed_test_paths = set()
            for item in fixed_split['test_split']:
                p = item['file_path']
                if data_root and not os.path.isabs(p):
                    p = os.path.normpath(os.path.join(data_root, p))
                fixed_test_paths.add(p)
            test_data = [item for item in data_list if item['image'] in fixed_test_paths]
            remaining_data = [item for item in data_list if item['image'] not in fixed_test_paths]
            remaining_labels = [item['label'] for item in remaining_data]
            if len(test_data) >= max(1, int(len(data_list) * test_ratio)):
                print(f"Using existing fixed test split from {fixed_split_path}")
                train_val_data, train_val_labels = remaining_data, remaining_labels
            else:
                print(
                    f"[WARN] Fixed split at {fixed_split_path} matches only {len(test_data)} of the "
                    f"{len(data_list)} subjects in the current dataset, fewer than the "
                    f"{max(1, int(len(data_list) * test_ratio))} required, so it cannot be used as-is.\n"
                    f"       This is expected when the bundled split is the de-identified edition while "
                    f"your data uses local filenames: the shipped split records which subjects formed the "
                    f"test set, but its pseudonymised names cannot be joined to renamed image files.\n"
                    f"       A new split will be generated with random_state=42 and written under "
                    f"output_dir. The bundled file will NOT be overwritten."
                )
                fixed_split = None

        if fixed_split is None:
            from sklearn.model_selection import train_test_split
            train_val_data, test_data, train_val_labels, test_labels = train_test_split(
                data_list, all_labels, test_size=test_ratio, stratify=all_labels, random_state=42
            )
            # Write the regenerated split under output_dir, never back over the bundled artifact.
            # Overwriting it would replace the de-identified split with local subject identifiers
            # and silently destroy the published test-set definition.
            regenerated_split_path = os.path.join(output_dir, "fixed_data_split_regenerated.json")
            save_fixed_data_split(
                test_split=test_data, train_ratio=train_ratio, val_ratio=val_ratio,
                test_ratio=test_ratio, random_state=42, file_path=regenerated_split_path,
            )
            print(f"Regenerated fixed test split saved to: {regenerated_split_path}")
            val_ratio_adj = val_ratio / (1 - test_ratio)
            train_data, val_data, train_labels, val_labels = train_test_split(
                train_val_data, train_val_labels, test_size=val_ratio_adj,
                stratify=train_val_labels, random_state=42
            )
        else:
            from sklearn.model_selection import train_test_split
            val_ratio_adj = val_ratio / (1 - test_ratio)
            train_data, val_data, train_labels, val_labels = train_test_split(
                train_val_data, train_val_labels, test_size=val_ratio_adj,
                stratify=train_val_labels, random_state=42
            )

        # 3. Data transforms
        train_transforms, val_transforms = get_data_transforms(augmentation_intensity, a_max=a_max)
        test_transforms = val_transforms

        # 4. Data loaders
        from monai.data import Dataset
        train_ds = Dataset(data=train_data, transform=train_transforms)
        val_ds = Dataset(data=val_data, transform=val_transforms)
        test_ds = Dataset(data=test_data, transform=test_transforms)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
            pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False,
            worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(GLOBAL_SEED)
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False,
            worker_init_fn=worker_init_fn
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False,
            worker_init_fn=worker_init_fn
        )

        # 5. Class weights
        class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * NUM_CLASSES
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)

        # 6. Model
        model = get_model(model_name, NUM_CLASSES, device, dropout_rate, attention_type=attention_type)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {model_name}, Params: {total_params:,}, Attention: {attention_type}")

        # 7. Loss / optimizer
        loss_func = get_loss_function(loss_name, weight=class_weights_tensor).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4,
            betas=(0.9, 0.999), eps=1e-8
        )

        # 8. Scheduler
        scheduler = get_scheduler(scheduler_name, optimizer, num_epochs, len(train_loader))

        # 9. Resuming: detect an existing checkpoint and restore the training state
        best_auc, best_f1, best_acc = 0.0, 0.0, 0.0
        best_auc_path = best_f1_path = best_acc_path = None
        patience_counter = 0
        start_epoch = 0

        # Prefer the overwriting "latest" checkpoint (loses at most 1 epoch); otherwise use the
        # historical checkpoint with the largest epoch
        latest_ckpt = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
        if not os.path.exists(latest_ckpt):
            latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt and os.path.exists(latest_ckpt):
            ckpt = safe_torch_load(latest_ckpt, map_location=device)
            model.base_model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if scheduler and ckpt.get('scheduler_state_dict'):
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = int(ckpt.get('epoch', 0))
            extra = ckpt.get('extra_state', {}) or {}
            best_auc = extra.get('best_auc', 0.0)
            best_f1 = extra.get('best_f1', 0.0)
            best_acc = extra.get('best_acc', 0.0)
            patience_counter = extra.get('patience_counter', 0)
            # Restore the best checkpoint paths (used later to delete the old best checkpoints)
            for f in os.listdir(checkpoint_dir):
                fp = os.path.join(checkpoint_dir, f)
                if f.endswith('_best_auc.pth'):
                    best_auc_path = fp
                elif f.endswith('_best_f1.pth'):
                    best_f1_path = fp
                elif f.endswith('_best_acc.pth'):
                    best_acc_path = fp
            print(f"[Resume] resuming training from epoch {start_epoch} "
                  f"(best_auc={best_auc:.4f}, best_f1={best_f1:.4f}, best_acc={best_acc:.4f})")

        # 10. Training loop
        for epoch in range(start_epoch, num_epochs):
            train_loss = train_epoch(
                model, train_loader, optimizer, loss_func, device, scheduler,
                grad_accum_steps, writer=writer, epoch=epoch
            )
            val_metrics = validate_epoch(
                model, val_loader, loss_func, device, NUM_CLASSES, writer=writer, epoch=epoch, tag="Val"
            )

            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['auc'])

            current_auc = val_metrics['auc']
            current_f1 = val_metrics['f1_macro']
            current_acc = val_metrics['accuracy']

            improved = False
            if current_auc > best_auc:
                best_auc = current_auc
                improved = True
                if best_auc_path and os.path.exists(best_auc_path):
                    os.remove(best_auc_path)
                best_auc_path = save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, checkpoint_dir, val_metrics,
                    config=config, tag='best_auc'
                )
            if current_f1 > best_f1:
                best_f1 = current_f1
                improved = True
                if best_f1_path and os.path.exists(best_f1_path):
                    os.remove(best_f1_path)
                best_f1_path = save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, checkpoint_dir, val_metrics,
                    config=config, tag='best_f1'
                )
            if current_acc > best_acc:
                best_acc = current_acc
                if best_acc_path and os.path.exists(best_acc_path):
                    os.remove(best_acc_path)
                best_acc_path = save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, checkpoint_dir, val_metrics,
                    config=config, tag='best_acc'
                )

            if improved:
                patience_counter = 0
            else:
                patience_counter += 1

            # Save the overwriting "latest" checkpoint every epoch (with resumable state, loses at
            # most 1 epoch)
            extra_state = {
                'best_auc': best_auc,
                'best_f1': best_f1,
                'best_acc': best_acc,
                'patience_counter': patience_counter,
            }
            save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_dir,
                            val_metrics, config=config, tag='latest', extra_state=extra_state)

            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_dir,
                                val_metrics, config=config)

            if epoch % 20 == 0 or epoch == num_epochs - 1:
                print(
                    f"Epoch {epoch+1}/{num_epochs} | Loss={train_loss:.4f} | Val AUC={current_auc:.4f} | "
                    f"F1={current_f1:.4f} | Acc={current_acc:.4f} | "
                    f"Best(AUC={best_auc:.4f}, F1={best_f1:.4f}, Acc={best_acc:.4f})"
                )

        # 10. Save results
        final_metrics = validate_epoch(model, val_loader, loss_func, device, NUM_CLASSES)
        result = {
            'run_id': run_id,
            'config': config,
            'best_auc': best_auc,
            'best_f1': best_f1,
            'best_acc': best_acc,
            'final_auc': final_metrics['auc'],
            'final_f1': final_metrics['f1_macro'],
            'final_acc': final_metrics['accuracy'],
            'total_params': total_params,
            'run_dir': run_dir,
            'status': 'completed'
        }
        result_df = pd.DataFrame([result])
        result_df.to_csv(os.path.join(run_dir, "training_result.csv"), index=False)
        append_global_training_result(result, csv_path=global_csv)

        print(f"Run {run_id} completed. Best AUC: {best_auc:.4f}, F1: {best_f1:.4f}, Acc: {best_acc:.4f}")
        return result

    except Exception as e:
        print(f"Run {run_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return {'run_id': run_id, 'config': config, 'status': 'failed', 'error': str(e)}

    finally:
        if writer is not None:
            writer.close()
        try:
            del model, optimizer, scheduler
            del train_loader, val_loader, test_loader
            del train_ds, val_ds, test_ds
        except Exception:
            pass
        import gc
        gc.collect()


def save_grid_search_results(results, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    flattened_results = []
    for result in results:
        flat_result = {
            'run_id': result.get('run_id', ''),
            'status': result.get('status', ''),
            'error': result.get('error', ''),
            'best_auc': result.get('best_auc', 0),
            'best_f1': result.get('best_f1', 0),
            'best_acc': result.get('best_acc', 0),
            'final_auc': result.get('final_auc', 0),
            'final_f1': result.get('final_f1', 0),
            'final_acc': result.get('final_acc', 0),
            'total_params': result.get('total_params', 0),
            'run_dir': result.get('run_dir', '')
        }
        config = result.get('config', {})
        for key, value in config.items():
            flat_result[f'config_{key}'] = value
        flattened_results.append(flat_result)
    pd.DataFrame(flattened_results).to_csv(output_path, index=False, encoding='utf-8-sig')
    return output_path


def run_grid_search(max_runs=None, resume_from=0, cfg=None, epochs=None):
    output_dir = cfg["output_dir"]
    combinations = generate_hyperparameter_combinations(epochs=epochs)
    if max_runs:
        combinations = combinations[:max_runs]

    # Automatically skip completed combinations: read all_training_results.csv and collect the
    # run_ids with status=completed
    global_csv = os.path.join(output_dir, "all_training_results.csv")
    completed_ids = set()
    if os.path.exists(global_csv):
        try:
            done_df = pd.read_csv(global_csv, encoding='utf-8-sig')
            if 'run_id' in done_df.columns and 'status' in done_df.columns:
                completed_ids = set(done_df.loc[done_df['status'] == 'completed', 'run_id'].tolist())
        except Exception as e:
            print(f"[WARN] failed to read {global_csv}; not skipping automatically: {e}")

    # resume_from acts as a lower threshold (skip run_id <= resume_from), unioned with the
    # automatic skips
    skip_below = resume_from
    n_skip = 0
    print(f"{len(combinations)} combinations in total; {len(completed_ids)} completed combinations "
          f"skipped automatically; combinations with run_id <= {skip_below} will also be skipped")

    results = []
    for run_id, config in enumerate(combinations, 1):
        if run_id <= skip_below:
            continue
        if run_id in completed_ids:
            print(f"Skipping completed combination Run {run_id}")
            n_skip += 1
            continue
        result = run_single_training(config, run_id, cfg)
        results.append(result)
        if run_id % 10 == 0:
            save_grid_search_results(results, f"grid_search_results_checkpoint_{run_id}.csv", output_dir)

    if results:
        save_grid_search_results(results, "grid_search_final_results.csv", output_dir)
    print(f"Completed {len(results)} new combinations this run; skipped {n_skip} completed combinations")
    return results


def main():
    parser = argparse.ArgumentParser(description="Grid-search training")
    parser.add_argument("--max_runs", type=int, default=None, help="Maximum number of runs")
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Skip combinations with run_id <= this value (optional; completed combinations are skipped automatically, so this usually need not be set manually)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (defaults to EPOCH_OPTIONS[0]=200)")
    args = parser.parse_args()

    cfg = load_config()
    os.makedirs(cfg["output_dir"], exist_ok=True)

    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_grid_search(max_runs=args.max_runs, resume_from=args.resume_from, cfg=cfg, epochs=args.epochs)


if __name__ == "__main__":
    main()
