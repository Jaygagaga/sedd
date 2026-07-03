import math
import os
import time

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import open_dict
import data
import losses
import sampling
import utils
from load_model import load_model_with_config
from model.ema import ExponentialMovingAverage


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@hydra.main(version_base=None, config_path="configs", config_name="sft")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.run.dir if hydra_cfg.mode == RunMode.RUN else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    utils.makedirs(work_dir)

    with open_dict(cfg):
        cfg.work_dir = work_dir
        cfg.wandb_name = os.path.basename(os.path.normpath(work_dir))

    logger = utils.get_logger(os.path.join(work_dir, "logs"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(cfg)

    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    metrics_dir = os.path.join(work_dir, "metrics")
    utils.makedirs(sample_dir)
    utils.makedirs(checkpoint_dir)
    utils.makedirs(os.path.dirname(checkpoint_meta_dir))
    utils.makedirs(metrics_dir)

    pretrained_path = getattr(cfg.training, "pretrained_path", "/root/workspace/sedd_small")
    score_model, graph, noise = load_model_with_config(pretrained_path, device, cfg)
    optimizer = losses.get_optimizer(cfg, score_model.parameters())
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    state = dict(
        optimizer=optimizer,
        scaler=scaler,
        model=score_model,
        noise=noise,
        ema=ema,
        step=0,
        epoch=0,
        best_eval_loss=float("inf"),
        best_epoch=-1,
        epochs_without_improvement=0,
        loss_history={"epochs": []},
    )
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)

    tokenizer = data.load_gpt2_tokenizer(getattr(cfg.data, "tokenizer_name_or_path", None))
    train_loader, valid_loader = data.get_dataloaders(cfg, distributed=False)
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum)
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum)

    train_size = data.get_sft_split_length(
        cfg.data.dataset_name,
        "train",
        cache_dir=cfg.data.cache_dir,
        seed=cfg.data.split_seed,
        local_path=getattr(cfg.data, "local_path", None),
    )
    valid_size = data.get_sft_split_length(
        cfg.data.dataset_name,
        "validation",
        cache_dir=cfg.data.cache_dir,
        seed=cfg.data.split_seed,
        local_path=getattr(cfg.data, "local_path", None),
    )
    train_micro_batches = max(1, math.ceil(train_size / cfg.training.batch_size) * cfg.training.accum)
    valid_batches_per_epoch = max(1, math.ceil(valid_size / cfg.eval.batch_size))
    sampling_shape = (1, cfg.model.length)
    loss_history_json = os.path.join(metrics_dir, "loss_history.json")
    loss_history_csv = os.path.join(metrics_dir, "loss_history.csv")
    loss_history_plot = os.path.join(metrics_dir, "loss_curve.png")

    for epoch in range(state["epoch"], cfg.training.epochs):
        state["epoch"] = epoch
        score_model.train()
        running_loss = 0.0
        running_logs = 0
        epoch_start = time.time()

        for batch_idx in range(train_micro_batches):
            batch = move_batch_to_device(next(train_loader), device)
            loss = train_step_fn(state, batch)
            if batch_idx % cfg.training.log_freq == 0 and loss is not None:
                logger.info(f"epoch={epoch} batch={batch_idx} step={state['step']} train_loss={loss.item():.6f}")

            if loss is not None:
                running_loss += loss.item()
                running_logs += 1

        avg_train_loss = running_loss / max(running_logs, 1)

        score_model.eval()
        eval_losses = []
        for _ in range(valid_batches_per_epoch):
            batch = move_batch_to_device(next(valid_loader), device)
            eval_loss = eval_step_fn(state, batch)
            eval_losses.append(eval_loss.item())
        avg_eval_loss = sum(eval_losses) / max(len(eval_losses), 1)
        prev_best_eval_loss = state["best_eval_loss"]

        logger.info(
            f"epoch={epoch} avg_train_loss={avg_train_loss:.6f} avg_eval_loss={avg_eval_loss:.6f} "
            f"elapsed={time.time() - epoch_start:.1f}s"
        )

        state["loss_history"]["epochs"].append(
            {
                "epoch": int(epoch),
                "step": int(state["step"]),
                "train_loss": float(avg_train_loss),
                "eval_loss": float(avg_eval_loss),
                "elapsed_seconds": float(time.time() - epoch_start),
            }
        )
        utils.save_json(loss_history_json, state["loss_history"])
        utils.save_loss_history_csv(loss_history_csv, state["loss_history"])
        utils.plot_loss_history(loss_history_plot, state["loss_history"])

        if avg_eval_loss < prev_best_eval_loss:
            state["best_eval_loss"] = avg_eval_loss
            state["best_epoch"] = epoch
            state["epochs_without_improvement"] = 0
            utils.save_checkpoint(os.path.join(checkpoint_dir, "best_checkpoint.pth"), state)
        else:
            state["epochs_without_improvement"] += 1

        utils.save_checkpoint(checkpoint_meta_dir, state)
        utils.save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"), state)

        sample_batch = move_batch_to_device(next(valid_loader), device)
        sampling_fn = sampling.get_conditional_sampling_fn(
            graph=graph,
            noise=noise,
            batch_dims=sampling_shape,
            predictor=cfg.sampling.predictor,
            steps=cfg.sampling.steps,
            prompt_clamp=getattr(cfg.sampling, "prompt_clamp", True),
            denoise=cfg.sampling.noise_removal,
            eps=1e-5,
            device=device,
        )
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sample_tokens = sampling_fn(
            score_model,
            sample_batch["input_ids"][:1],
            sample_batch["prompt_mask"][:1],
            sample_batch["target_mask"][:1],
        )
        ema.restore(score_model.parameters())
        sample_text = tokenizer.batch_decode(sample_tokens, skip_special_tokens=False)[0]
        with open(os.path.join(sample_dir, f"epoch_{epoch}.txt"), "w") as fout:
            fout.write(sample_text)

        if state["epochs_without_improvement"] >= cfg.training.early_stopping_patience:
            logger.info("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
