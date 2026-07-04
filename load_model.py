import os
import json
import torch
from model import SEDD
import utils
from model.ema import ExponentialMovingAverage
import graph_lib
import noise_lib

from omegaconf import OmegaConf


def _load_state_dict_from_file(path, device):
    loaded = torch.load(path, map_location=device, weights_only=False)
    if isinstance(loaded, dict) and 'model' in loaded:
        return loaded['model']
    return loaded


def _load_local_hf_state_dict(root_dir, device):
    bin_path = os.path.join(root_dir, "pytorch_model.bin")
    if os.path.exists(bin_path):
        return _load_state_dict_from_file(bin_path, device)

    safetensors_path = os.path.join(root_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Found model.safetensors but safetensors is not installed in the current environment."
            ) from exc
        return load_file(safetensors_path, device=str(device))

    raise FileNotFoundError(
        f"No supported Hugging Face weight file found in {root_dir}. "
        "Expected pytorch_model.bin or model.safetensors."
    )


def _load_checkpoint_payload(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def _load_training_checkpoint_into_model(model, checkpoint_path, device, prefer_ema=True):
    loaded = _load_checkpoint_payload(checkpoint_path, device)

    if isinstance(loaded, dict) and prefer_ema and "ema" in loaded:
        ema_state = loaded["ema"]
        ema = ExponentialMovingAverage(
            model.parameters(),
            decay=float(ema_state.get("decay", 0.9999)),
        )
        ema.load_state_dict(ema_state)
        ema.copy_to(model.parameters())
        return "ema"

    if isinstance(loaded, dict) and "model" in loaded:
        model.load_state_dict(loaded["model"], strict=False)
        return "model"

    if isinstance(loaded, dict):
        model.load_state_dict(loaded, strict=False)
        return "state_dict"

    raise TypeError(f"Unsupported checkpoint payload type in {checkpoint_path}: {type(loaded)!r}")

def load_model_hf(dir, device):
    if os.path.isdir(dir):
        config_path = os.path.join(dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json in local HF model directory: {dir}")
        with open(config_path, "r", encoding="utf-8") as fin:
            cfg = OmegaConf.create(json.load(fin))
        score_model = SEDD(cfg).to(device)
        state_dict = _load_local_hf_state_dict(dir, device)
        score_model.load_state_dict(state_dict, strict=False)
        graph = graph_lib.get_graph(cfg, device)
        noise = noise_lib.get_noise(cfg).to(device)
        return score_model, graph, noise

    score_model = SEDD.from_pretrained(dir).to(device)
    graph = graph_lib.get_graph(score_model.config, device)
    noise = noise_lib.get_noise(score_model.config).to(device)
    return score_model, graph, noise


def _looks_like_local_path(root_dir):
    return (
        os.path.isabs(root_dir)
        or root_dir.startswith(".")
        or root_dir.startswith("~")
        or os.sep in root_dir
    )


def load_model_local(root_dir, device):
    cfg = utils.load_hydra_config_from_run(root_dir)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    score_model = SEDD(cfg).to(device)

    ckpt_dir = os.path.join(root_dir, "checkpoints-meta", "checkpoint.pth")
    _load_training_checkpoint_into_model(score_model, ckpt_dir, device, prefer_ema=True)
    return score_model, graph, noise


def load_model_with_config(root_dir, device, cfg):
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    score_model = SEDD(cfg).to(device)

    if os.path.isfile(root_dir):
        loaded_kind = _load_training_checkpoint_into_model(score_model, root_dir, device, prefer_ema=True)
        print(f"Loaded {loaded_kind} weights from checkpoint file: {root_dir}")
        return score_model, graph, noise

    if os.path.isdir(root_dir):
        hf_weight_files = [
            os.path.join(root_dir, "pytorch_model.bin"),
            os.path.join(root_dir, "model.safetensors"),
        ]
        hf_config_path = os.path.join(root_dir, "config.json")
        if os.path.exists(hf_config_path) and any(os.path.exists(path) for path in hf_weight_files):
            state_dict = _load_local_hf_state_dict(root_dir, device)
            score_model.load_state_dict(state_dict, strict=False)
            return score_model, graph, noise

        best_ckpt = os.path.join(root_dir, "checkpoints", "best_checkpoint.pth")
        fallback_ckpt = os.path.join(root_dir, "checkpoints-meta", "checkpoint.pth")
        if os.path.exists(best_ckpt) or os.path.exists(fallback_ckpt):
            ckpt_dir = best_ckpt if os.path.exists(best_ckpt) else fallback_ckpt
            loaded_kind = _load_training_checkpoint_into_model(score_model, ckpt_dir, device, prefer_ema=True)
            print(f"Loaded {loaded_kind} weights from training run checkpoint: {ckpt_dir}")
            return score_model, graph, noise

        raise FileNotFoundError(
            f"Could not find a supported checkpoint inside directory: {root_dir}. "
            "Expected a Hugging Face export (config.json + pytorch_model.bin/model.safetensors) "
            "or a local experiment checkpoint under checkpoints/ or checkpoints-meta/."
        )

    pretrained_model = SEDD.from_pretrained(root_dir).to(device)
    score_model.load_state_dict(pretrained_model.state_dict(), strict=False)
    return score_model, graph, noise


def load_model(root_dir, device):
    if os.path.isfile(root_dir):
        raise ValueError(
            "load_model() cannot load a bare checkpoint file without config context. "
            "Pass a Hugging Face model directory/repo id, or a local run directory, "
            "or use load_model_with_config() when loading a .pth checkpoint."
        )

    if os.path.isdir(root_dir):
        hf_weight_files = [
            os.path.join(root_dir, "pytorch_model.bin"),
            os.path.join(root_dir, "model.safetensors"),
        ]
        hf_config_path = os.path.join(root_dir, "config.json")
        hydra_config_path = os.path.join(root_dir, ".hydra", "config.yaml")

        if os.path.exists(hf_config_path) and any(os.path.exists(path) for path in hf_weight_files):
            return load_model_hf(root_dir, device)
        if os.path.exists(hydra_config_path):
            return load_model_local(root_dir, device)

        raise FileNotFoundError(
            f"Could not find a supported model inside directory: {root_dir}. "
            "Expected either config.json + pytorch_model.bin/model.safetensors, "
            "or .hydra/config.yaml + checkpoints-meta/checkpoint.pth."
        )

    if _looks_like_local_path(root_dir):
        alt_root_dir = root_dir.replace("sedd-small", "sedd_small")
        hint = f" Did you mean '{alt_root_dir}'?" if alt_root_dir != root_dir else ""
        raise FileNotFoundError(f"Local model path does not exist: {root_dir}.{hint}")

    return load_model_hf(root_dir, device)
