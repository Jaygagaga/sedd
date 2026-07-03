import os
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

def load_model_hf(dir, device):
    score_model = SEDD.from_pretrained(dir).to(device)
    graph = graph_lib.get_graph(score_model.config, device)
    noise = noise_lib.get_noise(score_model.config).to(device)
    return score_model, graph, noise


def load_model_local(root_dir, device):
    cfg = utils.load_hydra_config_from_run(root_dir)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    score_model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    ckpt_dir = os.path.join(root_dir, "checkpoints-meta", "checkpoint.pth")
    loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)

    score_model.load_state_dict(loaded_state['model'])
    ema.load_state_dict(loaded_state['ema'])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    return score_model, graph, noise


def load_model_with_config(root_dir, device, cfg):
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    score_model = SEDD(cfg).to(device)

    if os.path.isfile(root_dir):
        state_dict = _load_state_dict_from_file(root_dir, device)
        score_model.load_state_dict(state_dict, strict=False)
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
            state_dict = _load_state_dict_from_file(ckpt_dir, device)
            score_model.load_state_dict(state_dict, strict=False)
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
    try:
        return load_model_hf(root_dir, device)
    except:
        return load_model_local(root_dir, device)
