# CARV: A Diagnostic Benchmark for Compositional Analogical Reasoning in Multimodal LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2603.27958-b31b1b.svg)](https://arxiv.org/abs/2603.27958)
[![COLM 2026](https://img.shields.io/badge/COLM-2026-blue.svg)](https://arxiv.org/abs/2603.27958)

**Authors:** Yongkang Du, Xiaohan Zou, Minhao Cheng, Lu Lin · Pennsylvania State University

## Overview

Existing visual analogy benchmarks are limited to single-step tasks: extract a transformation from one image pair and apply it to a query. CARV goes further by requiring models to **compose transformations from multiple image pairs** via logical set operations. Given *n* context pairs each showing a different atomic transformation, the model must synthesize a new rule through Union (∪), Intersection (∩), or Difference (\\) and apply it to a query image.

Images are drawn from a controlled visual domain defined by five properties — **subject**, **subject\_number**, **object**, **object\_color**, **spatial\_relation** — which allows precise isolation of reasoning failures from perceptual noise.

The benchmark spans **5,500 samples** across three task settings:

| Task | #Input pairs | #Atomic transforms | Count |
|---|---|---|---|
| Single-step | 1 | 2 | 500 |
| Compositional — Shared Source (SS) | 2 | 2, 3, 4 | 3,500 |
| Compositional — Different Source (DS) | 2 | 2 | 1,500 |
| **Total** | | | **5,500** |

**Shared Source (SS):** The two context pairs share the same source image as the query (*I<sub>q</sub> = I<sub>1</sub> = I<sub>2</sub>*). Lower contextual complexity; models can leverage shallow visual similarity.

**Different Source (DS):** The context pairs and the query all use distinct source images (*I<sub>q</sub> ≠ I<sub>1</sub> ≠ I<sub>2</sub>*). Requires decoupling the transformation rule from the specific visual context and applying it to a new image — a strictly harder abstraction demand.

**Complexity Scaling:** Extends the Shared Source setting by scaling the number of atomic transformations per pair from 2 to 3 and 4, testing robustness to increasing compositional depth.

## Dataset

Download from HuggingFace:

```bash
pip install huggingface-hub
huggingface-cli download duyongka/CARV --repo-type dataset --local-dir ./carv-data
```

Directory layout:

```
carv-data/
├── data/
│   ├── mix/          # images for the mix track (SS + DS, |T|=2)
│   └── large/        # images for the complexity scaling track (SS, |T|=3,4)
├── tasks/
│   ├── mix/
│   │   ├── union/
│   │   ├── intersection/
│   │   └── difference/
│   └── large/
│       ├── union/
│       └── intersection/
└── labels/
    ├── mix/
    └── large/
```

Each task JSON item contains:
- `context_image`: list of context image filenames
- `options`: list of candidate answer image filenames
- `answer_index`: ground-truth option index
- `changed_properties1`, `changed_properties2`: the property sets that differ in each input pair

## Setup

**Requirements:** Python 3.10, CUDA 12.1+

```bash
conda env create -f env.yml
conda activate visionllm
pip install -r requirements.txt
```

**API key configuration.** The inference script loads credentials from JSON config files. Create a `configs/` directory adjacent to this repo and add one file per provider:

```
configs/
├── openai_api_key_config.json       # GPT-4o
├── openai_api_key_config2.json      # GPT-5.1
├── geminipro_api_key_config.json    # Gemini-2.5 Pro
├── geminiflash_api_key_config.json  # Gemini-2.5 Flash
└── qwen32b_vl_api_key_config.json   # Qwen2.5-VL-32B (optional)
```

Each file follows this format (adapt fields per provider):

```json
{
    "api_key": "YOUR_API_KEY",
    "base_url": "https://api.openai.com/v1/",
    "model": "gpt-4o",
    "model_id": "gpt-4o",
    "max_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.95,
    "n": 1,
    "stop": []
}
```

Then point the code at your configs directory via the `CARV_CONFIG_DIR` environment variable (defaults to `../configs` relative to the repo root if unset):

```bash
export CARV_CONFIG_DIR=/path/to/your/configs
```

## Running Inference

```bash
cd src/inference
python analogy_composition.py \
    --task mix \
    --prompt step_by_step \
    --model gemini2.5flash \
    --number 500
```

**Arguments:**

| Argument | Default | Options |
|---|---|---|
| `--task` | `mix` | `mix` (SS+DS, \|T\|=2), `large` (SS, \|T\|=3,4), `single` (baseline) |
| `--prompt` | `step_by_step` | `step_by_step`, `direct` |
| `--model` | `gpt5` | `gpt`, `gpt5`, `gemini2.5pro`, `gemini2.5flash`, `qwen32b`, `qwen3`, `llama`, `o1` |
| `--number` | `500` | number of items per split |

Results are written to `outputs/{prompt}/{task}/{operation}/{model}_{difficulty}_{number}.jsonl`. Each item includes the model's free-text response and a `correctness_analysis` field from the GPT-4o automatic evaluator (98% agreement with human annotation on 200 samples).

## Failure Diagnosis

CARV includes a 4-stage diagnostic pipeline that pinpoints where a model first fails:

1. **Perception** — caption the context images and describe the visual transformation in each pair
2. **Decomposition** — break each transformation into atomic property-value changes (symbolic rules)
3. **Composition** — apply the set operation (∪ / ∩ / \\) to synthesize the target rule
4. **Application** — caption the query image after applying the composed rule

Run the diagnosis evaluator after inference:

```bash
cd src/evaluation

# Primary diagnosis on main inference outputs
python analysis_bn.py

# Diagnosis on the human-annotated subset (for kappa computation)
python analysis_annotation.py

# Diagnosis on hard-difficulty items only
python analysis_hard.py

# Diagnosis filtered by property combination
python analysis_property.py
```

Configure `task_category`, `models`, `operations`, and file paths in each script's `__main__` block before running.

## Inter-Annotator Agreement

```bash
cd annotation
python coefficient.py
```

Computes Cohen's Kappa between human failure-stage labels and GPT-4o diagnosis labels. Our evaluator achieves accuracy 0.82 and κ = 0.63 on 120 annotated samples.

- `coefficient_certain.py` — keeps only items where all 3 GPT runs agree (higher precision)
- `coefficient_vote.py` — majority vote across 3 runs (full coverage)

## Results

Performance on CARV compositional tasks (Accuracy %). SS = Shared Source, DS = Different Source.

| Model | Single | Union SS | Union DS | Inter. SS | Inter. DS | Diff. SS | Diff. DS |
|---|---|---|---|---|---|---|---|
| **Closed-source** | | | | | | | |
| GPT-5.1 | 81.2 | 64.6 | 51.0 | 75.0 | 73.0 | 59.0 | 62.4 |
| Gemini-2.5 Pro | 79.2 | 51.6 | 40.4 | 62.8 | 67.2 | 59.4 | 55.6 |
| Gemini-2.5 Flash | 75.0 | 49.4 | 37.4 | 60.2 | 64.2 | 41.4 | 33.2 |
| GPT-4o | 44.8 | 27.2 | 9.0 | 34.0 | 36.0 | 25.6 | 21.2 |
| **Open-source** | | | | | | | |
| Qwen3VL-8B-Thinking | 66.7 | 33.6 | 20.8 | 56.8 | 59.2 | 33.3 | 25.0 |
| Qwen2.5VL-32B | 39.0 | 18.6 | 6.2 | 30.4 | 32.8 | 25.4 | 15.0 |
| Qwen3VL-30B-A3B | 33.4 | 21.7 | 4.2 | 18.4 | 17.6 | 10.8 | 9.9 |
| Qwen2.5VL-7B | 2.4 | 9.0 | 2.2 | 10.0 | 8.6 | 3.0 | 4.2 |
| Human | — | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

Key findings: the primary bottleneck for closed-source models is **decomposition** (translating visual changes into symbolic rules); for open-source models it is **perception**. All models degrade significantly when moving from SS to DS, and as the number of atomic transformations increases.

## Citation

```bibtex
@article{du2026carv,
  title={CARV: A Diagnostic Benchmark for Compositional Analogical Reasoning in Multimodal LLMs},
  author={Du, Yongkang and Zou, Xiaohan and Cheng, Minhao and Lin, Lu},
  journal={arXiv preprint arXiv:2603.27958},
  year={2026}
}
```
