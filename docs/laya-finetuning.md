# Fine-tuning legal intent classification with Laya

This repository can use Laya as a System-1 decision model for the legal/advocacy intent classifier.

Laya is not a generative LLM. It receives a state plus a typed decision question and returns a choice/score/yes-no answer with probabilities. The upstream fine-tuning recipe uses RLCD: noisy-logit policy gradients with proper-scoring-rule reward plus soft cross-entropy, followed by temperature calibration.

## Dataset → Laya decision

Our JSONL is:

```json
{"text":"Quero revisar um contrato antes de assinar.","label":"CONTRATO","domain":"legal"}
```

For Laya, each example becomes one choice decision with the question: “Qual categoria jurídica representa melhor a intenção do usuário?” and the 12 legal categories as options.

Laya returns the complete probability distribution. This is important because the classifier should expose confidence, not only the argmax label.

## Convert the dataset

Use:

```bash
python scripts/legal_to_laya.py --input datasets/legal/train.jsonl --output datasets/legal/laya/train.jsonl
python scripts/legal_to_laya.py --input datasets/legal/eval.jsonl --output datasets/legal/laya/eval.jsonl
```

The generated records follow Laya's training shape: state, questions, and gold.

The first experiment uses one-hot gold probabilities. A stronger second experiment should replace them with human/teacher probability distributions, because the official RLCD recipe is designed around distributions rather than only hard labels.

## Local NVIDIA RTX 4070

A desktop RTX 4070 commonly has 12 GB VRAM, so memory is the main constraint. The English Laya checkpoint has about 421M parameters and the official recipe fine-tunes the encoder and decision head.

### Option A — WSL2

If WSL2 exposes the GPU correctly, use:

```bash
nvidia-smi
python3.11 -m venv .venv-laya
source .venv-laya/bin/activate
python -m pip install -U pip
python -m pip install torch
python -m pip install laya transformers datasets safetensors huggingface_hub pyarrow pandas scipy accelerate
```

Verify CUDA:

```python
import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
```

If `torch.cuda.is_available()` is false in WSL2, do not spend time trying to force the Laya process to use the GPU. Use the native Windows path below or Google Colab.

### Option B — native Windows + NVIDIA GPU

This path does **not** require WSL2. PyTorch supports native Windows with NVIDIA CUDA, and the Laya repository documents a native Windows PowerShell virtual environment. Install the CUDA-enabled PyTorch wheel selected for Windows from the official PyTorch installer instructions, before installing Laya.

First check the driver/GPU from PowerShell:

```powershell
nvidia-smi
py -3.11 --version
```

Create an isolated Python environment:

```powershell
py -3.11 -m venv .venv-laya
.\.venv-laya\Scripts\python.exe -m pip install -U pip
```

Install the **CUDA-enabled** PyTorch build. Use the current selector at https://pytorch.org/get-started/locally/ and choose:

```text
OS: Windows
Package: Pip
Language: Python
Compute Platform: CUDA
```

For example, if the selector gives a command using a `cuXXX` index, run that exact command:

```powershell
.\.venv-laya\Scripts\python.exe -m pip install torch --index-url <CUDA-index-from-PyTorch>
```

Do not copy a Linux/WSL CUDA command into PowerShell. The PyTorch wheel supplies the required CUDA runtime components; the NVIDIA driver remains required.

Verify that native Windows Python can see the GPU **before** installing/training Laya:

```powershell
.\.venv-laya\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else 0)"
```

The expected result is `CUDA: True` and your RTX 4070 name.

Then install Laya and the training dependencies:

```powershell
.\.venv-laya\Scripts\python.exe -m pip install laya
.\.venv-laya\Scripts\python.exe -m pip install transformers datasets safetensors huggingface_hub pyarrow pandas scipy accelerate
```

For the Python training implementation, use the upstream training repository:

```powershell
git clone https://github.com/NandhaKishorM/laya.git
cd laya
..\.venv-laya\Scripts\python.exe -m pip install -e .
```

Or, if you already cloned it, install it in editable mode from that checkout.

Download the base checkpoint:

```powershell
..\.venv-laya\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('convaiinnovations/laya', local_dir='../laya_base')"
```

From the dataset repository, generate the Laya records:

```powershell
cd ..\intent-datasets-4finetuning
.\.venv-laya\Scripts\python.exe scripts\legal_to_laya.py --input datasets\legal\train.jsonl --output datasets\legal\laya\train.jsonl
.\.venv-laya\Scripts\python.exe scripts\legal_to_laya.py --input datasets\legal\eval.jsonl --output datasets\legal\laya\eval.jsonl
```

If the virtual environment is outside the dataset repository, replace `\.venv-laya\Scripts\python.exe` with its absolute path.

### Recommended native-Windows settings for a 12 GB RTX 4070

Start conservatively:

```text
device                  cuda:0
micro-batch             1
gradient accumulation   32
epochs                  1-2
max_len                 512
head_max_len            192-256
gradient checkpointing  encoder + head
mixed precision         fp16
```

If the run is stable, increase micro-batch gradually. If it runs out of VRAM, reduce micro-batch first and/or reduce sequence lengths.

The upstream Laya project has a documented single-GPU fine-tuning example on a 16 GB GPU. That demonstrates the training approach on one GPU; a 12 GB RTX 4070 should be treated as the constrained configuration.

### Important Windows distinction

There are two different Laya repositories/roles:

- `receptron/laya`: Node.js/TypeScript runtime using ONNX.
- `NandhaKishorM/laya`: Python training/fine-tuning implementation.

For fine-tuning, use the Python training implementation. The Node.js package is useful after training for deploying the resulting decision model.

Do not turn this into a causal-LM `generate()` training loop. Laya is a decision model.

## Download the base checkpoint

```python
from huggingface_hub import snapshot_download
snapshot_download("convaiinnovations/laya", local_dir="./laya_base")
```

Then convert this repository's legal data:

```bash
python scripts/legal_to_laya.py --input datasets/legal/train.jsonl --output datasets/legal/laya/train.jsonl
```

The upstream preprocessing turns each case into tokenized items containing ids, markers, qtype, target and label.

For the first local run use:

```text
epochs=1
micro_batch=1
grad_accum=32
max_len=512
head_max_len=192
gradient_checkpointing=true
fp16=true
```

Only increase the workload after this completes without OOM.

## Google Colab

Create a Colab notebook with a GPU runtime and verify the actual GPU because hardware varies by session:

```python
!nvidia-smi
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Clone the training implementation:

```bash
!git clone https://github.com/NandhaKishorM/laya.git
%cd laya
!pip install -e .
```

Clone this dataset repository:

```bash
!git clone https://github.com/suissa/intent-datasets-4finetuning.git
```

Convert the legal data:

```bash
!python intent-datasets-4finetuning/scripts/legal_to_laya.py --input intent-datasets-4finetuning/datasets/legal/train.jsonl --output /content/legal_train_laya.jsonl
```

Download the checkpoint:

```python
from huggingface_hub import snapshot_download
snapshot_download("convaiinnovations/laya", local_dir="/content/laya_base")
```

Then adapt the upstream 2xT4 notebook to use /content/legal_train_laya.jsonl and the corresponding held-out legal evaluation data.

Colab sessions are ephemeral. Save checkpoints/reports to Drive if needed:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Use a destination such as /content/drive/MyDrive/laya/legal/ for the final checkpoint and benchmark report.

## What should be trained

The first legal model should remain a single choice decision:

```text
Human request
    ↓
Laya: Which legal intent/category?
    ↓
probabilities
    ↓
BehaviorID
    ↓
Skill
```

Do not put Skill execution instructions into the classification labels. The classifier learns what the human is asking; the Skill layer remains responsible for execution/specification.

## Evaluation and confidence

Record for every example: predicted_label, confidence, full_probability_distribution and correct.

Recommended metrics: accuracy, macro-F1, confusion matrix, Brier score, ECE, confidence on correct predictions, confidence on incorrect predictions, accuracy at confidence thresholds and out-of-domain rejection rate.

Temperature scaling should use a calibration split, while the final claim should be measured on a separate untouched evaluation split. Temperature scaling changes confidence, not the argmax label.

The useful output is not only CONTRATO; it is the distribution, for example:

```text
CONTRATO     0.91
CONSULTA     0.05
PROCESSO     0.02
...
```

That distribution can later drive Human-in-the-Loop decisions.

## Recommended experiment sequence

1. Run zero-shot Laya on datasets/legal/eval.jsonl.
2. Record accuracy, macro-F1 and confidence.
3. Fine-tune for one epoch.
4. Calibrate on a separate calibration split.
5. Evaluate on the untouched eval split.
6. Add hard negatives/boundary cases.
7. Fine-tune again.
8. Compare confusion matrices and confidence calibration.
9. Only then increase dataset size and training time.

The current legal dataset is a small synthetic pilot. It is an engineering baseline, not evidence of production-level legal classification quality.

## References

- Laya runtime: https://github.com/receptron/laya
- Laya training implementation: https://github.com/NandhaKishorM/laya
- Base checkpoint: https://huggingface.co/convaiinnovations/laya
- Fine-tuning notebook: https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
- Fine-tuning guide: https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md
- Single-GPU 16 GB example: https://github.com/NandhaKishorM/laya/blob/main/docs/finetune_browser_agent.md
- PyTorch Windows installation: https://pytorch.org/get-started/locally/
