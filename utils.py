import os
import json
import logging
import torch
from omegaconf import OmegaConf, open_dict


def load_hydra_config_from_run(load_dir):
    cfg_path = os.path.join(load_dir, ".hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg


def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        if 'optimizer' in state and 'optimizer' in loaded_state:
            state['optimizer'].load_state_dict(loaded_state['optimizer'])

        model = state.get('model')
        if model is not None and 'model' in loaded_state:
            model_to_load = model.module if hasattr(model, "module") else model
            model_to_load.load_state_dict(loaded_state['model'], strict=False)

        if 'ema' in state and 'ema' in loaded_state:
            state['ema'].load_state_dict(loaded_state['ema'])

        if 'scaler' in state and 'scaler' in loaded_state:
            state['scaler'].load_state_dict(loaded_state['scaler'])

        for key, value in loaded_state.items():
            if key not in {'optimizer', 'model', 'ema', 'scaler'} and key in state:
                state[key] = value
        return state


def save_checkpoint(ckpt_dir, state):
    model = state.get('model')
    model_to_save = model.module if hasattr(model, "module") else model

    saved_state = {}
    if 'optimizer' in state:
        saved_state['optimizer'] = state['optimizer'].state_dict()
    if model_to_save is not None:
        saved_state['model'] = model_to_save.state_dict()
    if 'ema' in state:
        saved_state['ema'] = state['ema'].state_dict()
    if 'scaler' in state:
        saved_state['scaler'] = state['scaler'].state_dict()

    for key, value in state.items():
        if key not in {'optimizer', 'model', 'ema', 'scaler', 'noise'}:
            saved_state[key] = value
    torch.save(saved_state, ckpt_dir)


def save_json(path, payload):
    with open(path, "w") as fout:
        json.dump(payload, fout, indent=2)


def save_loss_history_csv(path, history):
    rows = history.get("epochs", [])
    with open(path, "w") as fout:
        fout.write("epoch,train_loss,eval_loss,step,elapsed_seconds\n")
        for row in rows:
            fout.write(
                f"{row['epoch']},{row['train_loss']:.10f},{row['eval_loss']:.10f},"
                f"{row['step']},{row['elapsed_seconds']:.4f}\n"
            )


def plot_loss_history(path, history):
    import matplotlib

    mpl_config_dir = os.environ.get("MPLCONFIGDIR")
    if not mpl_config_dir:
        mpl_config_dir = os.path.join(os.path.dirname(path), ".mplconfig")
        os.environ["MPLCONFIGDIR"] = mpl_config_dir
    os.makedirs(mpl_config_dir, exist_ok=True)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = history.get("epochs", [])
    if not rows:
        return

    epochs = [row["epoch"] for row in rows]
    train_losses = [row["train_loss"] for row in rows]
    eval_losses = [row["eval_loss"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, eval_losses, marker="o", label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
