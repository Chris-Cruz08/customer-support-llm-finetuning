# Customer Support LLM Fine-Tuning with QLoRA and Ragas Evaluation

Fine-tuning Qwen2.5-1.5B-Instruct with QLoRA on real customer support data, with a Ragas evaluation pipeline that shows whether the fine-tune actually helped, not just that it runs.

## Problem Statement

Generic instruction-tuned LLMs answer customer support questions in a generic way: verbose, sometimes off-target, and not aligned with how a real support team actually communicates. Companies that want an LLM-powered support assistant either pay for expensive proprietary APIs at scale, or they need a smaller, specialized, self-hosted model that reliably matches their support style. This project builds the second option: taking a small, efficient open model and specializing it for customer support intent handling using parameter-efficient fine-tuning, on hardware most people already have.

## Architecture

```
+---------------------------+
|  Bitext Customer Support  |
|  Dataset (26,872 examples,|
|  27 intents)              |
+-------------+-------------+
              |
              v
+---------------------------+
|  Preprocessing             |
|  (ChatML formatting,       |
|   train/eval split)        |
+-------------+-------------+
              |
              v
+---------------------------+
|  QLoRA Fine-Tuning          |
|  Qwen2.5-1.5B-Instruct      |
|  4-bit NF4 + LoRA adapters  |
|  (RTX 2060, 6GB VRAM)       |
+-------------+-------------+
              |
      +-------+-------+
      |               |
      v               v
+-----------+   +-------------+
| Base Model |   | Fine-tuned  |
| (unchanged)|   | Model       |
+-----+-----+   +------+------+
      |                |
      +-------+--------+
              |
              v
+---------------------------+
|  Ragas Evaluation           |
|  faithfulness,              |
|  context precision,         |
|  hallucination rate         |
|  (local LLM judge)          |
+-------------+-------------+
              |
      +-------+-------+
      |               |
      v               v
+-----------+   +-------------+
| MLflow    |   | Gradio App  |
| Tracking  |   | (side by    |
|           |   |  side view) |
+-----------+   +------+------+
                       |
              +--------+--------+
              |                 |
              v                 v
      +-------------+   +----------------+
      | Docker       |   | HuggingFace    |
      | (local GPU)  |   | Spaces (ZeroGPU)|
      +-------------+   +----------------+
```

## Tech Stack

| Component | Tool / Version |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct |
| Fine-tuning method | QLoRA (4-bit NF4 quantization plus LoRA adapters) |
| Training libraries | transformers 4.44.2, peft 0.12.0, bitsandbytes 0.43.3, accelerate 0.34.2 |
| Dataset | Bitext Customer Support LLM Chatbot Training Dataset (HuggingFace) |
| Experiment tracking | MLflow 2.16.2 |
| Evaluation | Ragas 0.1.20, local HuggingFace model as judge |
| Demo UI | Gradio 4.44.1 |
| Hardware | NVIDIA RTX 2060 (6GB VRAM), Windows, PyCharm |
| Containerization | Docker with NVIDIA Container Toolkit (WSL2 backend) |
| Live deployment | HuggingFace Spaces (ZeroGPU) |

## Why Fine-Tuning and Not RAG

RAG makes sense when a model needs access to information that changes often or is too large to bake into its weights, like live inventory or account-specific records. That's not the problem here. The Bitext dataset represents a fixed set of about 27 support intents with fairly stable response patterns: the tone, the structure, the way escalations are phrased. The model doesn't need to look anything up. It needs to internalize how this company talks, consistently, with low latency and no retrieval step in the way.

Fine-tuning fits that better. It bakes the support team's voice and structure directly into the model's weights, so every response is on-brand and fast, without a vector store, a retrieval step, or context budget spent on retrieved documents. A production system would probably combine both approaches: RAG for account-specific or frequently changing facts, fine-tuning for consistent tone and structure. But for the narrow problem this project targets, fine-tuning alone is the more direct choice.

## Why Qwen2.5-1.5B-Instruct

This project originally targeted Phi-3 Mini (3.8B), chosen for its strong instruction-following at a size that, on paper, fits in 6GB of VRAM with 4-bit quantization. In practice, sustained QLoRA training at that size pushed the RTX 2060 into heavy thermal throttling (observed 83°C, power draw capped well below its rated wattage, roughly 10x slower than expected). That's a real hardware limit, not a config mistake.

Rather than fight the hardware, the project moved to Qwen2.5-1.5B-Instruct: about 2.5 times fewer parameters, a smaller memory footprint (1.48GB versus 2.05GB in 4-bit), and enough headroom to train at a sustainable pace without overheating. This is also a legitimate size and performance tradeoff, not just a workaround. For a narrow, templated task like intent-based customer support responses, a smaller model has plenty of room to specialize well, and the eval results below support that: the faithfulness gap between base and fine-tuned Qwen is large and consistent, which shows there was real room for the model to learn the target behavior.

## Training Details

**LoRA configuration:**

| Setting | Value | Why |
|---|---|---|
| Rank (r) | 16 | A common middle ground for LoRA. High enough to give the adapters real capacity to learn the support-response style, low enough to keep the adapter small and fast to train on consumer hardware. |
| Alpha | 32 | Set to 2x the rank, a standard default that scales the adapter's effective learning strength without needing a separate tuning pass. |
| Dropout | 0.05 | A small amount of regularization to reduce overfitting risk, given the dataset is fairly repetitive per intent. |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Covers both the attention layers and the MLP layers, so the adapters can adjust how the model attends to input and how it transforms that information, not just one or the other. |

This gave 18,464,768 trainable parameters out of 1,562,179,072 total, about 1.18%. That's the point of LoRA: the base model stays frozen and untouched, and only this small set of adapter weights gets updated.

**Training hyperparameters:**

| Setting | Value | Why |
|---|---|---|
| Batch size per device | 2 | Balanced against available VRAM. Small enough to fit comfortably, large enough to not waste time on excessive gradient accumulation steps. |
| Gradient accumulation steps | 2 | Combined with batch size 2, gives an effective batch size of 4, a reasonable size for stable gradients without needing more VRAM than available. |
| Epochs | 1 | Given the size of the training subsample and how repetitive the dataset is per intent, one pass was enough to bring training and validation loss down substantially and consistently. More epochs are listed as a next step below. |
| Learning rate | 2e-4 | A standard starting point for LoRA fine-tuning, high enough to make real progress in a single epoch, low enough to avoid destabilizing training. |
| Optimizer | adamw_bnb_8bit | An 8-bit AdamW optimizer that keeps optimizer states compressed in memory, cutting VRAM usage without hurting training quality, which matters when working with a 6GB GPU. |
| Precision | fp16 mixed precision | Halves memory usage for activations and gradients compared to full precision, with negligible quality loss for this task. |
| Max sequence length | 512 | Long enough to cover the vast majority of Bitext's instruction/response pairs without truncation, short enough to keep memory and compute cost down. |
| Warmup steps | 30 | A short warmup period so the learning rate ramps up gradually at the start rather than applying full-strength updates immediately. |
| Max gradient norm | 0.3 | Gradient clipping to prevent any single batch from causing a destabilizing update, a common safety measure for LoRA fine-tuning. |
| Gradient checkpointing | Off | Normally used to save memory by recomputing activations during the backward pass instead of storing them, at the cost of speed. With this smaller model there was enough VRAM headroom to skip it and train faster. |
| Training examples | 12,000 | A subsample of the full 24,184-example set, drawn evenly across all 27 intents (about 440 examples each), chosen as a time tradeoff. See "What I'd improve" below. |
| Total training steps | 3,000 | Determined by the training set size and effective batch size. |

**Loss curve** (logged every 300 steps):

| Step | Training Loss | Validation Loss |
|---|---|---|
| 300 | 0.7197 | 0.7330 |
| 600 | 0.6757 | 0.6685 |
| 900 | 0.6068 | 0.6416 |
| 1200 | 0.6413 | 0.6223 |
| 1500 | 0.6331 | 0.6050 |
| 1800 | 0.5951 | 0.5911 |
| 2100 | 0.6198 | 0.5805 |
| 2400 | 0.6073 | 0.5720 |
| 2700 | 0.5483 | 0.5640 |
| 3000 | 0.5543 | 0.5602 |

Final training loss: 0.6319.

Training loss bounces around a little step to step, which is normal for a noisy per-logging-window average, but validation loss drops steadily and smoothly from 0.73 down to 0.56 over the course of training, staying close to training loss the whole way through. That's a good sign: the model was generalizing to unseen examples, not just memorizing the training set.

## Evaluation Results

### What is Ragas, and what did we actually measure

Ragas is an evaluation library originally built for RAG pipelines, where it checks whether a chatbot's answer is actually supported by whatever documents it retrieved. It works by using a separate LLM as a judge: that judge model reads the question, the generated answer, and the reference content, then scores how well the answer holds up against that reference.

This project doesn't use retrieval, since it's a fine-tuning project, not RAG. So the reference answer from the dataset was used as the "ground truth" the model should stay consistent with, and Ragas scored each model's generated response against that reference. This was run twice: once for the base model's answers, once for the fine-tuned model's answers, so the two could be compared directly. The judge model was Qwen2.5-1.5B-Instruct running locally, so the whole evaluation is free and self-hosted, no paid API calls.

Evaluated on a held-out sample of 50 examples from the Bitext dataset that the model never saw during training.

| Metric | Base Model | Fine-tuned Model | Change |
|---|---|---|---|
| Faithfulness | 0.4578 | 0.6601 | +0.2023 (44% relative improvement) |
| Context Precision | 0.9000 | 0.9000 | No change |
| Hallucination Rate (1 minus faithfulness) | 0.5422 | 0.3399 | -0.2023 (37% relative reduction) |

**What each metric means here:**

- **Faithfulness** checks whether the model's answer actually matches what the reference response says, rather than drifting into something generic or unrelated. A base instruction-tuned model tends to answer in a technically reasonable but generic way, sometimes even misreading the intent entirely (in testing, the base model once responded to "can I speak to a human representative" as if being asked to place a phone call itself). The fine-tuned model's answers line up with the expected support response much more closely, which is what this metric is designed to catch. Going from 0.46 to 0.66 is a real, meaningful shift, not noise.
- **Hallucination rate** isn't a metric Ragas ships directly. It's calculated here as 1 minus faithfulness, which is a standard way to express the same signal in the opposite direction: how often the model says something not backed up by the reference. Fine-tuning cut this from about 54% to about 34%.
- **Context precision stayed flat at 0.90 for both models**, and that's expected rather than a disappointing result. Since there's no retrieval step in this project, the "context" fed into this metric was the reference answer itself, for both the base and fine-tuned evaluation. That means this particular metric is mostly measuring the relevance of the reference data, not anything about the generated response, so it makes sense that it doesn't move between the two models. It's included here for completeness and to be upfront about what this metric can and can't tell you outside of an actual RAG setup.

**A few honest notes on methodology:**
- Evaluation ran on 50 examples, not the full ~2,688-example held-out set, because each Ragas metric call needs multiple LLM calls per example, and running a local (non-API) judge model made that expensive in time. Scaling this up is listed below.
- Training used a 12,000-example subsample of the full 24,184-example training set (drawn evenly across all 27 intents, about 440 examples each), as a time tradeoff, also discussed below.

## Live Demo

HuggingFace Spaces (ZeroGPU): https://huggingface.co/spaces/christoph047/customer-support-llm-comparison

Runs on free ZeroGPU hardware, so the first request might take a little longer while a GPU is allocated.

![Gradio comparison interface placeholder](docs/gradio_screenshot_placeholder.png)
*Screenshot: side-by-side comparison interface showing base vs. fine-tuned model responses.*

## Repository Structure

```
customer-support-llm-finetuning/
├── README.md
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   └── app.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_finetuning_qlora.ipynb
│   └── 03_evaluation_ragas.ipynb
├── space_app/
│   ├── app.py
│   ├── requirements.txt
│   └── qwen_qlora_adapter/
└── mlflow/
    └── mlruns/
```

Training and evaluation are implemented in the notebooks (`02_finetuning_qlora.ipynb` and `03_evaluation_ragas.ipynb`), not as standalone scripts. The Gradio app in `src/app.py` is the one piece meant to run outside a notebook.

## Setup and Installation

Requirements: Python 3.11, an NVIDIA GPU with CUDA 12.1 support, Windows, Linux, or macOS.

```bash
git clone https://github.com/Chris-Cruz08/customer-support-llm-finetuning
cd customer-support-llm-finetuning
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Running Training

Training lives in `notebooks/02_finetuning_qlora.ipynb`. Open it in Jupyter (through PyCharm, VS Code, or `jupyter notebook`) and run the cells in order. The notebook:
1. Loads and formats the Bitext dataset into Qwen's ChatML template
2. Loads Qwen2.5-1.5B-Instruct in 4-bit NF4 quantization
3. Attaches LoRA adapters (r=16, alpha=32, targeting the attention and MLP projection layers)
4. Trains with MLflow tracking on, logging loss curves and hyperparameters
5. Saves the resulting adapter to `qwen_qlora_adapter/`

To view training metrics, run `mlflow ui --backend-store-uri file:///./mlflow/mlruns` and open the local URL it prints.

## Running Evaluation

Evaluation lives in `notebooks/03_evaluation_ragas.ipynb`. It:
1. Loads both the base model and the fine-tuned (adapter-attached) model
2. Generates responses from both on a held-out sample
3. Runs Ragas metrics (faithfulness, context precision) using a local model as judge
4. Saves results to `final_comparison_results.json` and logs them to MLflow

## Running the Gradio App

Locally (GPU required):
```bash
python src/app.py
```
Opens at `http://127.0.0.1:7860`.

Via Docker:
```bash
docker build -t customer-support-llm .
docker run --rm --gpus all -p 7860:7860 customer-support-llm
```
Requires Docker Desktop with the WSL2 backend and NVIDIA Container Toolkit support on Windows, or nvidia-docker on Linux.

## What I'd Improve With More Time and Compute

- Train on the full 24,184-example dataset instead of the 12,000-example subsample used here. Given how templated this dataset is, I'd expect a modest further improvement in faithfulness rather than a dramatic one, but it's the natural next step with more time.
- Scale Ragas evaluation up from 50 examples to the full held-out set, or at least a much larger sample, ideally with a stronger judge model for higher-quality scoring.
- Train for more than one epoch with proper overfitting checks. A few fine-tuned responses came out a bit too enthusiastic in tone, which more epochs with validation-based early stopping could smooth out.
- Add a RAG layer on top of the fine-tuned model for account-specific or frequently changing information, like order status, combining fine-tuning's consistent tone with RAG's up-to-date facts.
- Expand the Ragas metrics used beyond faithfulness and context precision, for example answer relevancy or semantic similarity to the reference, for a fuller evaluation picture.

## License

MIT
