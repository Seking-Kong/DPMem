<div align="center">

# DPMem: Training-Free Dual-loop Memory Compression for Streaming Video Understanding**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Qwen3-VL](https://img.shields.io/badge/Backbone-Qwen3--VL-7c3aed.svg)](https://github.com/QwenLM/Qwen3-VL)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Paper coming soon

</div>

> DPMem is a lightweight, training-free framework for streaming video understanding. It combines short-term visual redundancy compression with recurrent semantic memory so that pretrained video MLLMs can retain recent visual details and long-range temporal context under a bounded memory budget.

<p align="center">
  <img src="assets/framework.png" alt="DPMem framework overview" width="100%">
</p>

DPMem combines short-term visual redundancy compression with continuous semantic consolidation to maintain fine-grained visual evidence and long-range temporal context during streaming inference.

## Highlights

- **Training-free integration.** DPMem works with pretrained video MLLMs without additional finetuning.
- **Short-term visual memory.** Short-Term Window Redundancy Compression removes redundant spatial tokens across adjacent frames while preserving temporal boundaries.
- **Long-term semantic memory.** Continuous Semantic Consolidation converts compressed windows into a discrete and structured text sequence that is updated throughout the stream.
- **Online response control.** The current visual window, consolidated history, and incoming query are jointly used to answer immediately or wait for additional evidence.
- **Online and offline evaluation.** The released scripts cover StreamingBench and OVO-Bench. Offline benchmarks follow the evaluation protocol of lmms-eval v0.7.1.


## Repository layout

```text
DPMem/
├── assets/                         # README figures
├── evaluation/
│   └── lmms_eval/                  # Overlay files for lmms-eval v0.7.1
├── models/
│   ├── qwen2_5_vl/                # Qwen2.5-VL model adapter
│   └── qwen3_vl/                  # Qwen3-VL model adapter and token compression
├── scripts/
│   ├── streamingbench_dpmem.py    # StreamingBench inference and evaluation
│   ├── ovobench_dpmem.py          # OVO-Bench inference and evaluation
│   ├── run_streamingbench.sh      # StreamingBench launcher
│   └── run_ovobench.sh            # OVO-Bench launcher
├── tools/
│   └── compare_cuda_memory.py     # Compare memory records from two JSONL files
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

The following environment matches the environment used to run the released code.

```bash
conda create -n dpmem python=3.10 -y
conda activate dpmem

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

DPMem uses FFmpeg for video decoding. Install it through the operating system or Conda.

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Flash Attention is recommended. Install a wheel that matches Python 3.10, PyTorch 2.7, and the CUDA runtime in your environment. A compatible source build can also be installed as follows.

```bash
python -m pip install flash-attn==2.8.2 --no-build-isolation
```

Verify the core imports after installation.

```bash
python -c "import torch, transformers, accelerate; print(torch.__version__, transformers.__version__, accelerate.__version__)"
python -c "from models.qwen3_vl import Qwen3VLForConditionalGeneration; print('DPMem import OK')"
```

## Model preparation

The default checkpoint is [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). The launchers accept either this Hugging Face identifier or a local checkpoint path through `QWEN3_VL_CKPT_PATH`.

```bash
export QWEN3_VL_CKPT_PATH=Qwen/Qwen3-VL-8B-Instruct
```

## Data preparation

Download [StreamingBench](https://github.com/THUNLP-MT/StreamingBench) and [OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench) from their official sources. The default paths used by the launchers are shown below.

```text
data/
├── StreamingBench/
│   ├── StreamingBench/
│   │   └── Real_Time_Visual_Understanding.csv
│   └── Real-Time Visual Understanding/
└── OVO-Bench/
    ├── ovo_bench_new.json
    └── src_videos/
```

Custom locations can be provided without editing the source code.

```bash
export STREAMINGBENCH_TASK_CSV=data/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv
export STREAMINGBENCH_VIDEO_DIR="data/StreamingBench/Real-Time Visual Understanding"

export OVOBENCH_TASK_JSON=data/OVO-Bench/ovo_bench_new.json
export OVOBENCH_VIDEO_DIR=data/OVO-Bench/src_videos
```

## Online evaluation

### StreamingBench

```bash
bash scripts/run_streamingbench.sh
```

The reported peak StreamingBench setting can be reproduced by changing the sampling configuration through environment variables.

```bash
FPS=2 FRAMES_PER_CHUNK=40 bash scripts/run_streamingbench.sh
```

### OVO-Bench

```bash
bash scripts/run_ovobench.sh
```

The launchers expose the principal runtime configuration through environment variables. `NUM_CHUNKS` controls data parallel workers, and `GPUS_PER_WORKER` controls the devices assigned to each worker.

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_CHUNKS=2 bash scripts/run_streamingbench.sh
CUDA_VISIBLE_DEVICES=0,1 NUM_CHUNKS=2 bash scripts/run_ovobench.sh
```

## Offline evaluation

We evaluate MLVU and LongVideoBench using the task definitions and evaluation protocol provided by [`lmms-eval` v0.7.1](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.7.1), together with the DPMem model adapter and benchmark-specific task files released in this repository.

Install the exact upstream version first.

```bash
git clone --branch v0.7.1 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
python -m pip install -e .
cd ..
```

The files under [`evaluation/lmms_eval/`](evaluation/lmms_eval/) mirror their destinations under the upstream `lmms_eval/` package.

Copy the provided files to the corresponding folders in lmms-eval v0.7.1.

- Copy [`qwen3_vl.py`](evaluation/lmms_eval/models/simple/qwen3_vl.py) and [`modeling_qwen3_vl_dpmem.py`](evaluation/lmms_eval/models/simple/modeling_qwen3_vl_dpmem.py) to `lmms_eval/models/simple/`.
- Copy [`mcq_extract.py`](evaluation/lmms_eval/tasks/_task_utils/mcq_extract.py) to `lmms_eval/tasks/_task_utils/`.
- Copy MLVU [`utils.py`](evaluation/lmms_eval/tasks/mlvu/utils.py) and [`mlvu_dev.yaml`](evaluation/lmms_eval/tasks/mlvu/mlvu_dev.yaml) to `lmms_eval/tasks/mlvu/`.
- Copy LongVideoBench [`utils.py`](evaluation/lmms_eval/tasks/longvideobench/utils.py), [`longvideobench_val_i.yaml`](evaluation/lmms_eval/tasks/longvideobench/longvideobench_val_i.yaml), and [`longvideobench_val_v.yaml`](evaluation/lmms_eval/tasks/longvideobench/longvideobench_val_v.yaml) to `lmms_eval/tasks/longvideobench/`.

From the DPMem repository root, copy the complete overlay into the cloned lmms-eval repository.

```bash
export LMMS_EVAL_DIR=/path/to/lmms-eval
cp -r evaluation/lmms_eval/* "$LMMS_EVAL_DIR/lmms_eval/"
```

The released YAML files use paths relative to the lmms-eval repository root. Place the benchmark metadata under the following layout, or edit the YAML files for an equivalent relative layout.

```text
lmms-eval/
└── data/
    ├── MLVU/
    │   └── json/
    │       ├── 1_plotQA.json
    │       ├── 2_needle.json
    │       ├── 3_ego.json
    │       ├── 4_count.json
    │       ├── 5_order.json
    │       ├── 6_anomaly_reco.json
    │       └── 7_topic_reasoning.json
    └── LongVideoBench/
        └── lvb_val.json
```

Run lmms-eval from its repository root so that these relative paths resolve correctly. The following command shows the DPMem entry point. Set `drop_threshold` and the remaining model arguments to the configuration reported for the benchmark being reproduced.

```bash
cd "$LMMS_EVAL_DIR"
python -m lmms_eval \
  --model qwen3_vl \
  --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,drop_method=pixel,drop_threshold=0.1,drop_absolute=True,frames_per_chunk=20" \
  --tasks mlvu_dev,longvideobench_val_v \
  --batch_size 1 \
  --log_samples \
  --output_path ./outputs/dpmem_offline
```

## Results

<p align="center">
  <img src="assets/streamingbench.png" alt="StreamingBench results" width="100%">
</p>
<p align="center"><strong>StreamingBench</strong></p>

<p align="center">
  <img src="assets/ovobench.png" alt="OVO-Bench results" width="100%">
</p>
<p align="center"><strong>OVO-Bench</strong></p>

## Citation

The paper and BibTeX entry will be added after the public paper release.

## Acknowledgements

This repository builds on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [StreamingBench](https://github.com/THUNLP-MT/StreamingBench), [OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench), [TimeChat-Online](https://github.com/yaolinli/TimeChat-Online), and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). We thank the authors for releasing their models, code, and evaluation tools.

## License

This project is released under the [Apache License 2.0](LICENSE). Please also follow the licenses of the upstream models and evaluation datasets.
