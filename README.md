# LLM-FI — Language Model Fault Injection at Scale

Hands-on tutorial activity for fault injection (FI) into Large Language Model inference.
This guide walks through running FI experiments across different tasks, models, fault models,
and generation settings. Both **GPU** and **CPU** versions are provided.

Repository: https://github.com/pipijing13/LLM-FI

---

## Setup (10 min)

```
python -m venv llmfi
source llmfi/bin/activate
```

Clone the repository:

```
git clone https://github.com/pipijing13/LLM-FI.git
cd LLM-FI
```

We provide both a GPU and a CPU (with smaller model, Qwen2.5-0.5B-Instruct) version. Pick the one that matches your hardware.

| Version | GPU memory | System memory | Disk space |
|---------|-----------|---------------|------------|
| **GPU** | 20 GB     | 8 GB          | 30 GB      |
| **CPU** | —         | 8 GB          | 10 GB      |


### GPU version

Install PyTorch with CUDA 11.8, then the remaining dependencies:

```
python -m pip install torch==2.5.0+cu118 torchvision==0.20.0+cu118 torchaudio==2.5.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

### CPU version

Install the CPU build of PyTorch, then the remaining dependencies:

```
python -m pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements_cpu.txt
```


---

## Overview

Fault injection is controlled by a small set of command-line flags, shared across all scripts:

* `--fault_mode` — where faults are injected (see [Part 3](#part-3-fault-models-2-min))
* `--model` — target model (GPU version only; choices differ per task)
* `--task` — target dataset (multiple-choice script only)
* `--generation_mode` — decoding strategy for generative tasks (translation / summarization)
* `--num_trials` — number of bit-flip trials per input sample (default 500–1000)

---

## Part 1: Fault Injection across Tasks and Models (10 min)

The `basic/` directory contains one FI script per task type.

| Task | Script | Models (`--model`) |
|------|--------|--------------------|
| Multiple Choice | `multichoiceFI.py` | `qwen`, `llama3`, `falcon` |
| Math Solving | `gsm8kFI.py` | `falcon` (default), `qwen` |
| Q&A | `squadFI.py` | `qwen` (default), `llama3`, `falcon` |
| Translation | `wmtFI.py` | `alma` (default), `qwen`, `llama2` |
| Summarization | `xlsumFI.py` | `summarizer` (default), `llama3`, `qwen` |

Model name reference:
`qwen` = Qwen2.5-7B-Instruct, `llama3` = Llama-3.1-8B-Instruct, `falcon` = Falcon3-7B-Instruct,
`llama2` = Llama2-7B, `alma` = ALMA-7B, `summarizer` = Llama-3.1-8B-Summarizer.

For the CPU version example, we use Qwen2.5-0.5B-Instruct model, and there is no model selection.

For the multiple-choice script, `--task` selects the benchmark:
`mmlu` (tinyMMLU), `arc` (tinyArc / AI2_ARC), `hella` (tinyHellaSwag), `wino` (tinyWinoGrande),
`truth` (tinyTruthfulQA).

1. Move into the `basic/` directory.
   ```
   cd basic
   ```

2. Run a multiple-choice experiment.
   ```
   python multichoiceFI.py --fault_mode neuron --task mmlu --model qwen --num_trials 1000
   ```


3. Try the other task scripts. A few examples (GPU version):
   ```
   python gsm8kFI.py  --fault_mode neuron --model falcon  --num_trials 1000
   python squadFI.py  --fault_mode neuron --model llama3  --num_trials 1000
   python wmtFI.py    --fault_mode neuron --model alma    --num_trials 1000
   python xlsumFI.py  --fault_mode neuron --model summarizer --num_trials 1000
   ```
---

## Part 2: Examining the FI Output (10 min)

Each run creates an output folder named after the model, task, and fault mode, and prints a
summary to the console.

* `multichoiceFI.py` → `{model}{task}_{fault_mode}` (e.g. `qwenmmlu_neuron`)
* `gsm8kFI.py` → `gsm8kFI_{fault_mode}_{model}` (e.g. `gsm8kFI_weight_qwen`)

1. Run a short experiment so it finishes quickly.

   **GPU version:**
   ```
   cd basic
   python multichoiceFI.py --fault_mode neuron --task mmlu --model qwen --num_trials 5
   ```

   **CPU version:**
   ```
   python multichoiceFI_cpu.py --fault_mode neuron --task mmlu --num_trials 2
   ```

2. The console prints a summary once all trials complete. For the multiple-choice script it
   reports the clean baseline accuracy and the average accuracy under fault injection:
   ```
   --- Final results ---
   Baseline accuracy: 0.7229
   Average bit flip accuracy: 0.7229 (± 0.0000)
   Average evaluation time: 31.18 seconds
   ```

3. Inspect the output folder. Per-trial JSON files and aggregated summaries are written there.
   ```
   ls qwenmmlu_neuron/
   # baseline_results.json  summary_results.json  all_results_final.json  trial_000_results.json ...
   cat qwenmmlu_neuron/summary_results.json
   ```

4. Repeat for the math task. The `weight` fault mode and the per-answer logs make it easy to
   see which generations changed.

   **GPU version:**
   ```
   python gsm8kFI.py --fault_mode weight --model qwen --num_trials 2
   ```

   **CPU version:**
   ```
   python gsm8kFI_cpu.py --fault_mode weight --num_trials 2
   ```
   ```
   ls gsm8kFI_weight_qwen/
   # all_answers.txt  different_answers.txt
   ```

---

## Part 3: Chain-of-Thought (2 min)

`nocotFI.py` (in the repository root) runs the math task **without** chain-of-thought prompting,
so you can compare CoT vs. no-CoT resilience against the GSM8K result from Part 2.
Models: `falcon` (default), `qwen`.
```
cd ..
python nocotFI.py --fault_mode neuron --model qwen --num_trials 1000
```

---

## Part 4: Fault Models (5 min)

The `--fault_mode` flag selects how and where faults are injected. The three modes map onto
two fault classes:

| `--fault_mode` | Fault class | Description |
|----------------|-------------|-------------|
| `single` | Computational | Single-bit flip on a neuron |
| `neuron` | Computational | Double-bit flip on a neuron |
| `weight` | Memory | Double-bit flip on a model weight |

Run the same task under different fault modes to compare resilience:
```
cd basic
python gsm8kFI.py --fault_mode single --model falcon --num_trials 1000
python gsm8kFI.py --fault_mode neuron --model falcon --num_trials 1000
python gsm8kFI.py --fault_mode weight --model falcon --num_trials 1000
```

---


## Part 5: Generation Mode — Beam vs. Greedy (2 min)

The generative scripts (`wmtFI.py` translation, `xlsumFI.py` summarization) accept a
`--generation_mode` flag:

| `--generation_mode` | Description |
|---------------------|-------------|
| `greedy` (default) | Greedy decoding |
| `beam` | Beam search |

Compare the two decoding strategies under the same fault injection.
```
cd basic
python wmtFI.py --fault_mode neuron --model alma --generation_mode greedy --num_trials 1000
python wmtFI.py --fault_mode neuron --model alma --generation_mode beam   --num_trials 1000
```

---

## Part 6: Mixture-of-Experts (MoE) (2 min)

The `moe/` directory compares a dense model against an MoE model of comparable architecture.

* Dense: Llama-3.2-3B-Instruct → `dense*FI.py`
* MoE: Llama-3.2-8X3B-MOE-18.4B → `moe*FI.py`

Available tasks: `mmlu`, `arc` (ARC_AI2), `squad` (Q&A), `wmt` (Translation), e.g.
`densemmluFI.py` / `moemmluFI.py`, `densearcFI.py` / `moearcFI.py`, and so on.
These scripts take only `--num_trials`.

```
cd moe
python densemmluFI.py --num_trials 1000
python moemmluFI.py   --num_trials 1000
```

---

## Part 7: Trace Analysis — Output Types (5 min)

As a final step, it helps to see *how* a generation goes wrong, not just whether accuracy
dropped. The `trace/` directory analyzes pre-collected GSM8K traces and classifies each
faulty generation by output type (e.g. the subtly wrong outputs, or the distorted outputs) across the three fault models, for Falcon3-7B and Qwen2.5-7B. It runs on
bundled traces, so no GPU is needed.

```
cd trace
pip install matplotlib
python trace_analyze.py
```

The script writes two artifacts:

* `output_type.txt` — text summary (percentage of generations whose result changed vs.
  fell into an infinite loop)
* `figs/output_type.pdf` — bar chart of output types per model and fault model

```
cat output_type.txt
```

---

## Command Reference

```bash
python multichoiceFI.py --fault_mode neuron --model qwen --task mmlu --num_trials 1000
```

### `--fault_mode` (default: `weight`)
* `weight` — double-bit flip on model weights (memory fault)
* `neuron` — double-bit flip on neuron outputs (computational fault)
* `single` — single-bit flip on neuron outputs (computational fault)

### `--generation_mode` (default: `greedy`, generative scripts only)
* `greedy` — greedy decoding
* `beam` — beam search

### `--model` (GPU version only; choices depend on the script)
* `qwen` — Qwen2.5-7B-Instruct
* `llama3` — Llama-3.1-8B-Instruct
* `llama2` — Llama2-7B
* `falcon` — Falcon3-7B-Instruct
* `alma` — ALMA-7B
* `summarizer` — Llama-3.1-8B-Summarizer

### `--task` (multiple-choice script only)
* `mmlu`, `arc`, `hella`, `wino`, `truth`

### `--num_trials` (default: 500 / 1000)
Number of bit-flip trials per input sample.
