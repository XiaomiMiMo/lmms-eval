# The Evaluation Suite of Xiaomi MiMo-VL

To promote **rigorous**, **reproducible**, and **thinking-oriented** evaluation of Vision-Language Models (VLMs), we open-source our evaluation suite for for [**MiMo-VL**](https://github.com/XiaomiMiMo/MiMo-VL) and beyond.

Built on top of the excellent [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) framework, we introduce several improvements in model integration, evaluation protocol, and task coverage to better support the next generation of reasoning-capable VLMs.

## 📰 News

**[25/08/08]** We update our evaluation framework along with the release of [MiMo-VL-7B-SFT-2508](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-SFT-2508) and [MiMo-VL-7B-RL-2508](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL-2508). New features include: 
- Additional GUI action benchmarks AndroidControl and CAGUI (evaluated using `--model mimo_agent`)
- Additional evaluation benchmarks on video spatial reasoning (VSI-Bench), physics reasoning (PhysReason), multi-modal long context understanding (MMLongBench), multi-modal instruction following (MM-IFEval); 
- Enables no_think evaluation by adding model argument `disable_thinking_user=True`


## 🔧 Key Features

### 1. ⚙️ `MiVLLM`: A vLLM-based Model Wrapper for MiMo-VL
We introduce a new `MiVLLM` model class based on the original `VLLM` class in `lmms-eval`, which is tailored for [**MiMo-VL**](https://github.com/XiaomiMiMo/MiMo-VL). Compared to the original implementation, it:
* Greatly improves **data loading efficiency**
* Enables fine-grained control over **image and video preprocessing**

### 2. 🧠 Adaptation to Thinking VLMs

The original lmms-eval tasks were designed for **non-thinking** VLMs: they prompt directly for short answers and compare outputs without post-processing. We redesign this process to support **reasoning-intensive models**:

* Introduce a unified `\boxed{}` output format using the prompt: *Put your final answer in `\boxed{}`.*
* Extend `max_new_tokens` to **32768** to allow the model to reason before answering
* Automatically extract predictions from the final `\boxed{}` output


### 3. 📏 Refined Open-ended Evaluation Metrics

For open-ended tasks such as **DocVQA**, **InfoVQA**, **ChartQA**, and **OCRBench**, we calculate accuracy using **GPT-4o** as the evaluator. This improves the fidelity of evaluation for free-form answers and better reflects model capabilities.


### 4. 🧩 20+ New Tasks for Comprehensive Evaluation

We contribute **over 20 new evaluation tasks** covering:

* **General vision-language understanding**
* **Math and logic reasoning**
* **GUI understanding and grounding**
* **Video understanding and reasoning**

👉 A complete list of supported tasks is available [here](mimovl_docs/tasks.md).


## Usage

### Installation

```bash
git clone https://github.com/XiaomiMiMo/lmms-eval
cd lmms-eval
pip install -e . && pip uninstall -y opencv-python-headless
pip install -r requirements.txt
```

### Evaluation Script

```bash
bash mimovl_docs/eval_mimo_vl.sh
```

Reproduction of MiMo-VL-7B-SFT results in our technical report can be found [here](https://docs.google.com/spreadsheets/d/e/2PACX-1vTAJPLH8O5MH7yvdeCj_UTW1TodOLDEvl4TToh_U9K0GS7VFEDrmzC8UE29XPoKM72pgyRhxNgbuXut/pubhtml).




## Citations

```shell
@misc{coreteam2025mimovl,
      title={MiMo-VL Technical Report}, 
      author={{Xiaomi LLM-Core Team}},
      year={2025},
      url={https://github.com/XiaomiMiMo/MiMo-VL}, 
}

@misc{mimovleval2025,
    title={The Evaluation Suite of Xiaomi MiMo-VL},
    author={LLM-Core Xiaomi},
    year={2025},
    url={https://github.com/XiaomiMiMo/lmms-eval}
}
```
