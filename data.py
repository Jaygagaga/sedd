import os
import re
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from itertools import chain
import numpy as np
import torch
from typing import Dict, List

import urllib.request
import zipfile
import requests
import json
from datasets import Dataset

from torch.utils.data import DataLoader, DistributedSampler


SFT_DATASET_NAME = "simplescaling/s1K-1.1"
SFT_DEFAULT_SPLITS = {
    "train": (0, 800),
    "validation": (800, 900),
    "test": (900, 1000),
}

DEFAULT_TOKENIZER_CANDIDATES = (
    "/gpt_tokenizer",
    "/root/workspace/gpt_tokenizer",
    os.path.join(os.getcwd(), "gpt_tokenizer"),
)


def load_gpt2_tokenizer(tokenizer_name_or_path=None):
    candidates = []
    if tokenizer_name_or_path:
        candidates.append(tokenizer_name_or_path)
    candidates.extend(DEFAULT_TOKENIZER_CANDIDATES)
    candidates.append("gpt2")

    last_error = None
    for candidate in candidates:
        try:
            tokenizer = GPT2TokenizerFast.from_pretrained(candidate, local_files_only=candidate != "gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to load GPT-2 tokenizer from candidates: {candidates}") from last_error


def _format_sft_example(prompt, target):
    return f"Question:\n{prompt.strip()}\n\nSolution:\n{target.strip()}"


def _normalize_example_value(example, field, fallback_fields):
    if field in example and example[field] is not None:
        return example[field]

    for fallback in fallback_fields:
        if fallback in example and example[fallback] is not None:
            return example[fallback]

    available = ", ".join(example.keys())
    raise KeyError(f"Field '{field}' not found. Available fields: {available}")


def _extract_answer_text(text):
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text, flags=re.DOTALL)
    if boxed:
        return boxed[-1].strip()

    if "####" in text:
        tail = text.rsplit("####", 1)[-1].strip()
        if tail:
            return tail.splitlines()[0].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def _resolve_sft_data_files(local_path):
    if not local_path:
        return None

    if os.path.isfile(local_path):
        return {"train": local_path}

    if os.path.isdir(local_path):
        parquet_files = sorted(
            os.path.join(local_path, name)
            for name in os.listdir(local_path)
            if name.endswith(".parquet")
        )
        if parquet_files:
            return {"train": parquet_files}

    raise FileNotFoundError(f"SFT local data path not found or has no parquet files: {local_path}")


def _load_sft_dataset_source(dataset_name, cache_dir=None, local_path=None):
    if local_path:
        data_files = _resolve_sft_data_files(local_path)
        return load_dataset("parquet", data_files=data_files, cache_dir=cache_dir)

    if dataset_name and (dataset_name.endswith(".parquet") or os.path.exists(dataset_name)):
        data_files = _resolve_sft_data_files(dataset_name)
        return load_dataset("parquet", data_files=data_files, cache_dir=cache_dir)

    return load_dataset(dataset_name, cache_dir=cache_dir)


def get_sft_dataset(
    dataset_name,
    mode,
    tokenizer,
    prompt_field="question",
    target_field="solution",
    answer_field=None,
    cache_dir=None,
    local_path=None,
    max_length=512,
    seed=42,
):
    dataset = _load_sft_dataset_source(dataset_name, cache_dir=cache_dir, local_path=local_path)

    if mode in dataset:
        data = dataset[mode]
    elif mode == "validation" and "test" in dataset:
        data = dataset["test"]
    else:
        base_split = dataset["train"]
        shuffled = base_split.shuffle(seed=seed)
        if mode not in SFT_DEFAULT_SPLITS:
            raise ValueError(f"Unsupported SFT split: {mode}")
        start, end = SFT_DEFAULT_SPLITS[mode]
        end = min(end, len(shuffled))
        data = shuffled.select(range(start, end))

    eos_id = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess(example):
        prompt = _normalize_example_value(example, prompt_field, ("question", "prompt", "input"))
        target = _normalize_example_value(example, target_field, ("solution", "output", "response"))
        answer_text = (
            _normalize_example_value(example, answer_field, ("answer",))
            if answer_field is not None
            else example.get("answer")
        )
        formatted_prompt = f"Question:\n{prompt.strip()}\n\nSolution:\n"
        prompt_ids = tokenizer(formatted_prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target.strip(), add_special_tokens=False)["input_ids"]
        answer_text = answer_text if answer_text is not None else _extract_answer_text(target)

        if len(prompt_ids) >= max_length:
            prompt_ids = prompt_ids[: max_length - 1]

        max_target_len = max_length - len(prompt_ids) - 1
        max_target_len = max(max_target_len, 0)
        target_ids = target_ids[:max_target_len]

        input_ids = prompt_ids + target_ids + [eos_id]
        target_len = len(target_ids) + 1
        prompt_len = len(prompt_ids)

        prompt_mask = [1] * prompt_len + [0] * target_len
        target_mask = [0] * prompt_len + [1] * target_len

        return {
            "input_ids": input_ids,
            "prompt_mask": prompt_mask,
            "target_mask": target_mask,
            "prompt_len": prompt_len,
            "raw_target_text": target.strip(),
            "answer_text": str(answer_text).strip(),
            "full_text": _format_sft_example(prompt, target),
        }

    processed = data.map(preprocess, remove_columns=data.column_names, load_from_cache_file=True)
    processed = processed.with_format(
        "torch",
        columns=["input_ids", "prompt_mask", "target_mask", "prompt_len"],
        output_all_columns=True,
    )
    return processed


def get_sft_split_length(dataset_name, mode, cache_dir=None, seed=42, local_path=None):
    dataset = _load_sft_dataset_source(dataset_name, cache_dir=cache_dir, local_path=local_path)
    if mode in dataset:
        return len(dataset[mode])
    if mode == "validation" and "test" in dataset:
        return len(dataset["test"])
    if "train" not in dataset:
        raise ValueError(f"Dataset {dataset_name} has no train split for synthesized SFT split.")

    total = len(dataset["train"])
    start, end = SFT_DEFAULT_SPLITS[mode]
    return max(0, min(end, total) - min(start, total))


def collate_sft_batch(batch, pad_token_id):
    max_len = max(item["input_ids"].shape[0] for item in batch)

    def pad_1d(tensor, pad_value):
        out = torch.full((max_len,), pad_value, dtype=tensor.dtype)
        out[: tensor.shape[0]] = tensor
        return out

    collated = {
        "input_ids": torch.stack([pad_1d(item["input_ids"], pad_token_id) for item in batch]),
        "prompt_mask": torch.stack([pad_1d(item["prompt_mask"], 0) for item in batch]).bool(),
        "target_mask": torch.stack([pad_1d(item["target_mask"], 0) for item in batch]).bool(),
        "prompt_len": torch.stack([item["prompt_len"] for item in batch]),
        "raw_target_text": [item["raw_target_text"] for item in batch],
        "answer_text": [item["answer_text"] for item in batch],
        "full_text": [item["full_text"] for item in batch],
    }
    return collated


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def _build_loader_kwargs(config, sampler, shuffle, collate_fn=None):
    num_workers = int(getattr(config.data, "num_workers", 4))
    pin_memory = bool(getattr(config.data, "pin_memory", True)) and torch.cuda.is_available()
    persistent_workers = bool(getattr(config.data, "persistent_workers", True)) and num_workers > 0

    loader_kwargs = {
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": shuffle,
    }
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
    return loader_kwargs


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset


def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8):
    if name == "wikitext103":
        dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()
    else:
        dataset = load_dataset(name, cache_dir=cache_dir)

    if name == "lambada":
        data = dataset
    else:
        data = dataset[mode]

    if name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                 text[i] = detokenizer(t)
            return text
        return detok

    tokenizer = load_gpt2_tokenizer(getattr(config.data, "tokenizer_name_or_path", None))
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        if name == "ptb":
            text = example['sentence']
        else:
            text = example["text"]
        # print(list(example.keys()))
        # exit()
        
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(text, return_attention_mask=False)
        # add in EOS token following 
        # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
        for token in tokens['input_ids']:
            token.append(EOS)
        return tokens
    
    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    else:
        tokenized_dataset = tokenized_dataset.remove_columns('text')
    

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    chunked_dataset = chunked_dataset.with_format('torch')

    return chunked_dataset


def get_dataloaders(config, distributed=True):
    if getattr(config.data, "task_type", "lm") == "sft":
        tokenizer = load_gpt2_tokenizer(getattr(config.data, "tokenizer_name_or_path", None))

        train_set = get_sft_dataset(
            getattr(config.data, "dataset_name", SFT_DATASET_NAME),
            "train",
            tokenizer,
            prompt_field=getattr(config.data, "prompt_field", "question"),
            target_field=getattr(config.data, "target_field", "solution"),
            answer_field=getattr(config.data, "answer_field", None),
            cache_dir=config.data.cache_dir,
            local_path=getattr(config.data, "local_path", None),
            max_length=getattr(config.data, "max_length", config.model.length),
            seed=getattr(config.data, "split_seed", 42),
        )
        valid_set = get_sft_dataset(
            getattr(config.data, "dataset_name", SFT_DATASET_NAME),
            "validation",
            tokenizer,
            prompt_field=getattr(config.data, "prompt_field", "question"),
            target_field=getattr(config.data, "target_field", "solution"),
            answer_field=getattr(config.data, "answer_field", None),
            cache_dir=config.data.cache_dir,
            local_path=getattr(config.data, "local_path", None),
            max_length=getattr(config.data, "max_length", config.model.length),
            seed=getattr(config.data, "split_seed", 42),
        )

        if distributed:
            train_sampler = DistributedSampler(train_set)
            test_sampler = DistributedSampler(valid_set)
        else:
            train_sampler = None
            test_sampler = None

        collate_fn = lambda batch: collate_sft_batch(batch, tokenizer.pad_token_id)
        train_loader = cycle_loader(DataLoader(
            train_set,
            batch_size=config.training.batch_size,
            **_build_loader_kwargs(
                config,
                sampler=train_sampler,
                shuffle=(train_sampler is None),
                collate_fn=collate_fn,
            ),
        ))
        valid_loader = cycle_loader(DataLoader(
            valid_set,
            batch_size=config.eval.batch_size,
            **_build_loader_kwargs(
                config,
                sampler=test_sampler,
                shuffle=(test_sampler is None),
                collate_fn=collate_fn,
            ),
        ))
        return train_loader, valid_loader

    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")


    train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length)
    valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "text8" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        **_build_loader_kwargs(
            config,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
        ),
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        **_build_loader_kwargs(
            config,
            sampler=test_sampler,
            shuffle=(test_sampler is None),
        ),
    ))
    return train_loader, valid_loader
