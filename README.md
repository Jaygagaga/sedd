# Score Entropy Discrete Diffusion
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repo contains a PyTorch implementation for the paper [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
](https://arxiv.org/abs/2310.16834) by [Aaron Lou](https://aaronlou.com), [Chenlin Meng](https://cs.stanford.edu/~chenlin/) and [Stefano Ermon](https://cs.stanford.edu/~ermon/).

![cover](assets/main.gif)

## Design Choices

This codebase is built modularly to promote future research (as opposed to a more compact framework, which would be better for applications). The primary files are 

1. ```noise_lib.py```: the noise schedule
2. ```graph_lib```: the forward diffusion process
3. ```sampling.py```: the sampling strategies
4. ```model/```: the model architecture

## Installation

Simply run

```
conda env create -f environment.yml
```

which will create a ```sedd``` environment with packages installed. Note that this installs with CUDA 11.8, and different CUDA versions must be installed manually. The biggest factor is making sure that the ```torch``` and ```flash-attn``` packages use the same CUDA version (more found [here](https://github.com/Dao-AILab/flash-attention)).

## Working with Pretrained Models

### Download Models

Our pretrained models are hosted on huggingface ([small](https://huggingface.co/louaaron/sedd-small), [medium](https://huggingface.co/louaaron/sedd-medium)). However, models can also be loaded in locally (say after training). All functionality is found in ```load_model.py```.

```
# load in a pretrained model
pretrained_small_model, graph, noise = load_model("louaaron/sedd-small")
pretrained_medium_model, graph, noise = load_model("louaaron/sedd-medium")
# load in a local experiment
local_model, graph, noise = load_model("exp_local/experiment)
```

This loading gives the model, as well as the graph and noise (which are used for the loss/sampling setup).

### Run Sampling

We can run sampling using a command 

```
python run_sample.py --model_path MODEL_PATH --steps STEPS
```

We can also sample conditionally using

```
python run_sample_cond.py --model_path MODEL_PATH --step STEPS --prefix PREFIX --suffix SUFFIX
```

## Training New Models

### Run Training

We provide training code, which can be run with the command
```
python run_train.py
```
This creates a new directory `direc=exp_local/DATE/TIME` with the following structure (compatible with running sampling experiments locally)
```
├── direc
│   ├── .hydra
│   │   ├── config.yaml
│   │   ├── ...
│   ├── checkpoints
│   │   ├── checkpoint_*.pth
│   ├── checkpoints-meta
│   │   ├── checkpoint.pth
│   ├── samples
│   │   ├── iter_*
│   │   │   ├── sample_*.txt
│   ├── logs
```
Here, `checkpoints-meta` is used for reloading the run following interruptions, `samples` contains generated images as the run progresses, and `logs` contains the run output. Arguments can be added with `ARG_NAME=ARG_VALUE`, with important ones being:
```
ngpus                     the number of gpus to use in training (using pytorch DDP)
training.accum            number of accumulation steps, set to 1 for small and 2 for medium (assuming an 8x80GB node)
noise.type                one of geometric, loglinear 
graph.type                one of uniform, absorb
model                     one of small, medium
model.scale_by_sigma      set to False if graph.type=uniform (not yet configured)
```
Some example commands include
```
# training hyperparameters for SEDD absorb
python train.py noise_lib=loglinear graph.type=absorb model=medium training.accum=2
# training hyperparameters for SEDD uniform
python train.py noise_lib=geometric graph.type=uniform model=small model.scale_by_sigma=False
```

## Other Features

### SLURM compatibility

To train on slurm, simply run 
```
python train.py -m args
```

## Supervised Fine-Tuning for Conditional Generation

This fork includes a supervised fine-tuning path for adapting SEDD to question-to-solution generation, with [`simplescaling/s1K-1.1`](https://huggingface.co/datasets/simplescaling/s1K-1.1) as the default dataset. The goal is not to reproduce the original SEDD language-modeling benchmark, but to test whether a discrete diffusion LM can be adapted to conditional reasoning-style generation.

The SFT path is intentionally minimal:

1. Build an input sequence from a prompt field and a target field.
2. Keep the prompt prefix fixed.
3. Apply discrete diffusion only to the target suffix.
4. Train the SEDD ratio-score model with Score Entropy loss on target positions.
5. Evaluate generated solutions with lexical-overlap metrics and saved per-example generations.

### SFT Data Format

The default SFT dataset fields are:

```text
prompt_field = question
target_field = solution
answer_field = null
```

For each example, the model sees a formatted conditional sequence:

```text
Question:
{question}

Solution:
{solution}
```

The prompt part is used as conditioning context. The target part is where noise is applied and where the loss is computed. If the Hugging Face dataset only has a `train` split, the data loader creates deterministic train/validation/test splits using `data.split_seed`.

The main SFT settings live in `configs/sft.yaml`:

```yaml
data:
  task_type: sft
  dataset_name: simplescaling/s1K-1.1
  prompt_field: question
  target_field: solution
  max_length: 512
  split_seed: 42

training:
  target_only_diffusion: True

sampling:
  prompt_clamp: True
```

If the data is available locally, pass a local parquet file or parquet directory:

```bash
python train_sft.py data.local_path=/path/to/train-00000-of-00001.parquet
```

### Train SFT

Start with the default small debug run:

```bash
python train_sft.py
```

Useful medium-size training recipes:

```bash
# Medium model, 768-token context.
python train_sft.py \
  model=medium \
  model.length=768 \
  data.max_length=768 \
  training.batch_size=16 \
  hydra.run.dir=/root/shared-storage/sedd_runs/sft_medium_768_b16

# Medium model, 1024-token context.
python train_sft.py \
  model=medium \
  model.length=1024 \
  data.max_length=1024 \
  training.batch_size=8 \
  hydra.run.dir=/root/shared-storage/sedd_runs/sft_medium_1024_b8
```

The training script writes checkpoints and lightweight diagnostics under `hydra.run.dir`:

```text
checkpoints/best_checkpoint.pth
checkpoints/checkpoint_epoch_*.pth
checkpoints-meta/checkpoint.pth
metrics/loss_history.json
metrics/loss_history.csv
samples/epoch_*.txt
```

The best checkpoint is selected by validation loss.

### Evaluate SFT and Pretrained Baselines

Evaluate a pretrained SEDD baseline plus the latest SFT checkpoint under a run directory:

```bash
python eval_sft.py \
  eval.pretrained_model_path=louaaron/sedd-medium \
  eval.sft_search_root=/root/shared-storage/sedd_runs/sft_medium_1024_b8 \
  model=medium \
  model.length=1024 \
  data.max_length=1024 \
  sampling.steps=128 \
  hydra.run.dir=/root/shared-storage/sedd_runs/sft_medium_1024_b8_eval
```

This evaluates:

```text
pretrained_analytic
sft_analytic
sft_euler
```

To evaluate only the pretrained model, filter the model list:

```bash
python eval_sft.py \
  eval.models=pretrained_analytic \
  eval.pretrained_model_path=louaaron/sedd-medium \
  eval.sft_search_root=/root/shared-storage/sedd_runs/sft_medium_1024_b8 \
  model=medium \
  model.length=1024 \
  data.max_length=1024 \
  sampling.steps=128 \
  hydra.run.dir=/root/shared-storage/sedd_runs/pretrained_medium_1024_eval
```

`eval.sft_search_root` still needs to point at an existing SFT run because the evaluator resolves an SFT checkpoint before applying the optional model filter.

Evaluation writes:

```text
eval_outputs/metrics.json
eval_outputs/pretrained_analytic_generations.json
eval_outputs/pretrained_analytic_samples.txt
eval_outputs/sft_analytic_generations.json
eval_outputs/sft_analytic_samples.txt
eval_outputs/sft_euler_generations.json
eval_outputs/sft_euler_samples.txt
```

The `*_generations.json` files are the most useful for error analysis. Each record contains:

```text
index
generated_text
reference_text
pred_answer
ref_answer
rouge_l_f1
token_f1
exact_match
runtime_sec
```

### Metrics

The evaluator reports:

- `ROUGE-L F1`: longest-common-subsequence overlap between generated and reference solution.
- `token_f1`: bag-of-token overlap between generated and reference solution.
- `exact_match`: normalized final predicted answer exactly equals the reference answer.
- `avg_generated_length`: average generated length in tokenizer tokens.
- `avg_runtime_sec`: average wall-clock generation time per example.
- `num_examples`: number of evaluated test examples.

For reasoning-style data, `ROUGE-L` and `token_f1` are lexical-overlap indicators. They are useful for comparing runs, but they do not prove mathematical or scientific correctness. Always inspect `*_generations.json` when a run improves only slightly or generates much longer outputs.

### Comparing SFT Runs

A recommended comparison table is:

```text
pretrained SEDD medium, analytic
SFT SEDD medium, analytic
SFT SEDD medium, euler
SFT SEDD small, analytic
SFT SEDD small, euler
```

When reporting results, compare each SFT checkpoint to the pretrained baseline evaluated with the same context length, sampling steps, dataset split, and metric code. A useful interpretation pattern is:

```text
The SFT checkpoint improves lexical-overlap metrics, but exact match remains low.
The improvement should therefore be read as better topical/token overlap, not as reliable answer correctness.
```
## Citation
```
@article{lou2024discrete,
  title={Discrete diffusion modeling by estimating the ratios of the data distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```
## Acknowledgements

This repository builds heavily off of [score sde](https://github.com/yang-song/score_sde_pytorch), [plaid](https://github.com/igul222/plaid), and [DiT](https://github.com/facebookresearch/DiT).

