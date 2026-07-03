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
        max_length=getattr(cfg.data, "max_length", cfg.model.length),
        seed=getattr(cfg.data, "split_seed", 42),
    )
    collate_fn = lambda batch: data.collate_sft_batch(batch, tokenizer.pad_token_id)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model_specs = [
        ("pretrained_analytic", getattr(cfg.eval, "pretrained_model_path", "louaaron/sedd-small"), "analytic", False),
        ("sft_analytic", getattr(cfg.eval, "sft_model_path", cfg.work_dir), "analytic", True),
        ("sft_euler", getattr(cfg.eval, "sft_model_path", cfg.work_dir), "euler", True),
    ]

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

        metrics = {
            "rouge_l_f1": sum(rouge_scores) / max(len(rouge_scores), 1),
            "token_f1": sum(f1_scores) / max(len(f1_scores), 1),
            "exact_match": sum(em_scores) / max(len(em_scores), 1),
            "avg_generated_length": sum(lengths) / max(len(lengths), 1),
            "avg_runtime_sec": sum(runtimes) / max(len(runtimes), 1),
            "num_examples": len(records),
        }
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
