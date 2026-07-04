import json
import os
import re
import time
from collections import Counter

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
import data
import sampling
import utils
from load_model import load_model, load_model_with_config


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_answer(text):
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text, flags=re.DOTALL)
    if boxed:
        return boxed[-1].strip()

    if "####" in text:
        tail = text.rsplit("####", 1)[-1].strip()
        if tail:
            return tail.splitlines()[0].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def rouge_l_f1(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    dp = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i in range(1, len(pred_tokens) + 1):
        for j in range(1, len(ref_tokens) + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def trim_generated_text(full_text):
    if "Solution:" in full_text:
        return full_text.split("Solution:", 1)[-1].strip()
    return full_text.strip()


def _find_latest_epoch_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None

    latest = None
    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth$")
    for name in os.listdir(checkpoint_dir):
        match = pattern.fullmatch(name)
        if not match:
            continue
        epoch = int(match.group(1))
        path = os.path.join(checkpoint_dir, name)
        mtime = os.path.getmtime(path)
        candidate = (epoch, mtime, path)
        if latest is None or candidate > latest:
            latest = candidate
    return latest[-1] if latest is not None else None


def _resolve_checkpoint_in_run_dir(run_dir):
    if not os.path.isdir(run_dir):
        return None

    best_checkpoint = os.path.join(run_dir, "checkpoints", "best_checkpoint.pth")
    if os.path.exists(best_checkpoint):
        return best_checkpoint

    latest_epoch_checkpoint = _find_latest_epoch_checkpoint(os.path.join(run_dir, "checkpoints"))
    if latest_epoch_checkpoint is not None:
        return latest_epoch_checkpoint

    meta_checkpoint = os.path.join(run_dir, "checkpoints-meta", "checkpoint.pth")
    if os.path.exists(meta_checkpoint):
        return meta_checkpoint

    hf_bin = os.path.join(run_dir, "pytorch_model.bin")
    hf_safe = os.path.join(run_dir, "model.safetensors")
    if os.path.exists(os.path.join(run_dir, "config.json")) and (os.path.exists(hf_bin) or os.path.exists(hf_safe)):
        return run_dir

    return None


def _discover_latest_sft_checkpoint(search_root):
    if not os.path.exists(search_root):
        raise FileNotFoundError(f"SFT checkpoint path does not exist: {search_root}")

    if os.path.isfile(search_root):
        return search_root, os.path.dirname(search_root)

    direct_match = _resolve_checkpoint_in_run_dir(search_root)
    if direct_match is not None:
        return direct_match, search_root

    latest = None
    for root, _, _ in os.walk(search_root):
        resolved = _resolve_checkpoint_in_run_dir(root)
        if resolved is None:
            continue
        timestamp = os.path.getmtime(resolved if os.path.isfile(resolved) else root)
        candidate = (timestamp, resolved, root)
        if latest is None or candidate > latest:
            latest = candidate

    if latest is None:
        raise FileNotFoundError(
            f"No saved SFT checkpoint found under: {search_root}. "
            "Expected checkpoints-meta/checkpoint.pth or checkpoints/checkpoint_epoch_*.pth."
        )

    _, resolved_path, run_dir = latest
    return resolved_path, run_dir


def _normalize_model_filter(value):
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return set(parts) if parts else None
    if isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value if str(part).strip()]
        return set(parts) if parts else None
    return {str(value).strip()}


@hydra.main(version_base=None, config_path="configs", config_name="sft")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.run.dir if hydra_cfg.mode == RunMode.RUN else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    utils.makedirs(work_dir)
    logger = utils.get_logger(os.path.join(work_dir, "eval_logs"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = data.load_gpt2_tokenizer(getattr(cfg.data, "tokenizer_name_or_path", None))

    test_set = data.get_sft_dataset(
        getattr(cfg.data, "dataset_name", data.SFT_DATASET_NAME),
        "test",
        tokenizer,
        prompt_field=getattr(cfg.data, "prompt_field", "question"),
        target_field=getattr(cfg.data, "target_field", "solution"),
        answer_field=getattr(cfg.data, "answer_field", None),
        cache_dir=cfg.data.cache_dir,
        local_path=getattr(cfg.data, "local_path", None),
        max_length=getattr(cfg.data, "max_length", cfg.model.length),
        seed=getattr(cfg.data, "split_seed", 42),
    )
    collate_fn = lambda batch: data.collate_sft_batch(batch, tokenizer.pad_token_id)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    sft_search_root = getattr(cfg.eval, "sft_model_path", None) or getattr(cfg.eval, "sft_search_root", "exp_local/sft")
    sft_model_path, sft_run_dir = _discover_latest_sft_checkpoint(sft_search_root)
    logger.info(f"Resolved latest SFT checkpoint: {sft_model_path}")
    logger.info(f"Resolved SFT run directory: {sft_run_dir}")

    model_specs = [
        ("pretrained_analytic", getattr(cfg.eval, "pretrained_model_path", "/root/workspace/sedd_small"), "analytic", False),
        ("sft_analytic", sft_model_path, "analytic", True),
        ("sft_euler", sft_model_path, "euler", True),
    ]
    model_filter = _normalize_model_filter(getattr(cfg.eval, "models", None))
    if model_filter is not None:
        model_specs = [spec for spec in model_specs if spec[0] in model_filter]
        logger.info(f"Filtered eval models: {sorted(model_filter)}")
    if not model_specs:
        raise ValueError("No eval models selected. Check cfg.eval.models.")

    max_examples = getattr(cfg.eval, "max_examples", None)
    if max_examples is not None:
        max_examples = int(max_examples)
        logger.info(f"Debug eval enabled: max_examples={max_examples}")

    outputs_dir = os.path.join(work_dir, "eval_outputs")
    utils.makedirs(outputs_dir)
    results = {}

    for name, model_path, predictor, use_cfg_loader in model_specs:
        logger.info(f"Evaluating {name} from {model_path} with predictor={predictor}")
        if use_cfg_loader:
            model, graph, noise = load_model_with_config(model_path, device, cfg)
        else:
            model, graph, noise = load_model(model_path, device)
        model.eval()

        sampling_fn = sampling.get_conditional_sampling_fn(
            graph=graph,
            noise=noise,
            batch_dims=(1, cfg.model.length),
            predictor=predictor,
            steps=cfg.sampling.steps,
            prompt_clamp=getattr(cfg.sampling, "prompt_clamp", True),
            denoise=cfg.sampling.noise_removal,
            eps=1e-5,
            device=device,
        )

        records = []
        rouge_scores = []
        f1_scores = []
        em_scores = []
        lengths = []
        runtimes = []

        for idx, batch in enumerate(test_loader):
            if max_examples is not None and idx >= max_examples:
                break
            batch = move_batch_to_device(batch, device)
            start = time.time()
            generated_ids = sampling_fn(model, batch["input_ids"], batch["prompt_mask"], batch["target_mask"])
            elapsed = time.time() - start
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
            generated_text = trim_generated_text(decoded)
            reference_text = batch["raw_target_text"][0]
            pred_answer = extract_answer(generated_text)
            ref_answer = batch["answer_text"][0]

            rouge = rouge_l_f1(generated_text, reference_text)
            f1 = token_f1(generated_text, reference_text)
            em = float(normalize_text(pred_answer) == normalize_text(ref_answer))

            rouge_scores.append(rouge)
            f1_scores.append(f1)
            em_scores.append(em)
            lengths.append(len(tokenizer(generated_text, add_special_tokens=False)["input_ids"]))
            runtimes.append(elapsed)

            record = {
                "index": idx,
                "generated_text": generated_text,
                "reference_text": reference_text,
                "pred_answer": pred_answer,
                "ref_answer": ref_answer,
                "rouge_l_f1": rouge,
                "token_f1": f1,
                "exact_match": em,
                "runtime_sec": elapsed,
            }
            records.append(record)
            logger.info(
                f"{name} example={idx} em={em:.0f} rouge_l_f1={rouge:.4f} "
                f"token_f1={f1:.4f} runtime_sec={elapsed:.2f}"
            )

        metrics = {
            "rouge_l_f1": sum(rouge_scores) / max(len(rouge_scores), 1),
            "token_f1": sum(f1_scores) / max(len(f1_scores), 1),
            "exact_match": sum(em_scores) / max(len(em_scores), 1),
            "avg_generated_length": sum(lengths) / max(len(lengths), 1),
            "avg_runtime_sec": sum(runtimes) / max(len(runtimes), 1),
            "num_examples": len(records),
        }
        if name.startswith("sft_"):
            metrics["resolved_model_path"] = sft_model_path
            metrics["resolved_run_dir"] = sft_run_dir
        results[name] = metrics

        with open(os.path.join(outputs_dir, f"{name}_generations.json"), "w") as fout:
            json.dump(records, fout, indent=2, ensure_ascii=False)

        with open(os.path.join(outputs_dir, f"{name}_samples.txt"), "w") as fout:
            for record in records[:5]:
                fout.write(f"[{record['index']}]\n")
                fout.write(f"Generated:\n{record['generated_text']}\n\n")
                fout.write(f"Reference:\n{record['reference_text']}\n")
                fout.write("=" * 80 + "\n")

    with open(os.path.join(outputs_dir, "metrics.json"), "w") as fout:
        json.dump(results, fout, indent=2, ensure_ascii=False)
    logger.info(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
