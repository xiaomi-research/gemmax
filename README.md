<div align="center">

# GemmaX: Multilingual Translator based on Gemma Open Models

<img src='/images/gemmax.png' width='600' height='337'>

**🎮 [Try the live demo on HuggingFace](https://huggingface.co/spaces/xiaomi-research/milmmt-46-translation)**

</div>


## 📰 Updates

* **Feb. 12 2026**: The MiLMMT-v0.1 paper [Scaling Model and Data for Multilingual Machine Translation with Open Large Language Models](https://arxiv.org/abs/2602.11961) is available on ArXiv!
* **Jan. 23 2025**: The GemmaX2 paper [Multilingual Machine Translation with Open Large Language Models at Practical Scale: An Empirical Study](https://arxiv.org/abs/2502.02481) has been accepted at **NAACL 2025**!


## 📥 Models

Model checkpoints are released at huggingface:

> [!IMPORTANT]
> The `*-Pretrain` checkpoints are **NOT** translation models.

### GemmaX2-28

| Model | Description |
|-------|-------------|
| [GemmaX2-28-2B-Pretrain](https://huggingface.co/ModelSpace/GemmaX2-28-2B-Pretrain) | Continual pretraining of [Gemma2-2B](https://huggingface.co/google/gemma-2-2b). |
| [GemmaX2-28-2B-v0.1](https://huggingface.co/ModelSpace/GemmaX2-28-2B-v0.1) | Finetuned on GemmaX2-28-2B-Pretrain with translation instructions (v0.1). |
| [GemmaX2-28-2B-v0.2](https://huggingface.co/xiaomi-research/GemmaX2-28-2B-v0.2) | Finetuned on GemmaX2-28-2B-Pretrain with translation instructions (v0.2). |
| [GemmaX2-28-9B-Pretrain](https://huggingface.co/ModelSpace/GemmaX2-28-9B-Pretrain) | Continual pretraining of [Gemma2-9B](https://huggingface.co/google/gemma-2-9b). |
| [GemmaX2-28-9B-v0.1](https://huggingface.co/ModelSpace/GemmaX2-28-9B-v0.1) | Finetuned on GemmaX2-28-9B-Pretrain with translation instructions (v0.1). |
| [GemmaX2-28-9B-v0.2](https://huggingface.co/xiaomi-research/GemmaX2-28-9B-v0.2) | Finetuned on GemmaX2-28-9B-Pretrain with translation instructions (v0.2). |

### MiLMMT-46

| Model | Description |
|-------|-------------|
| [MiLMMT-46-1B-Pretrain](https://huggingface.co/xiaomi-research/MiLMMT-46-1B-Pretrain) | Continual pretraining of [Gemma3-1B](https://huggingface.co/google/gemma-3-1b-pt). |
| [MiLMMT-46-1B-v0.1](https://huggingface.co/xiaomi-research/MiLMMT-46-1B-v0.1) | Finetuned on MiLMMT-46-1B-Pretrain with translation instructions. |
| [MiLMMT-46-1B-v1.0](https://huggingface.co/xiaomi-research/MiLMMT-46-1B-v1.0) | Reinforcement learning and model merging on [MiLMMT-46-1B-v0.1](https://huggingface.co/xiaomi-research/MiLMMT-46-1B-v0.1). |
| [MiLMMT-46-4B-Pretrain](https://huggingface.co/xiaomi-research/MiLMMT-46-4B-Pretrain) | Continual pretraining of [Gemma3-4B](https://huggingface.co/google/gemma-3-4b-pt). |
| [MiLMMT-46-4B-v0.1](https://huggingface.co/xiaomi-research/MiLMMT-46-4B-v0.1) | Finetuned on MiLMMT-46-4B-Pretrain with translation instructions. |
| [MiLMMT-46-4B-v1.0](https://huggingface.co/xiaomi-research/MiLMMT-46-4B-v1.0) | Reinforcement learning and model merging on [MiLMMT-46-4B-v0.1](https://huggingface.co/xiaomi-research/MiLMMT-46-4B-v0.1). |
| [MiLMMT-46-12B-Pretrain](https://huggingface.co/xiaomi-research/MiLMMT-46-12B-Pretrain) | Continual pretraining of [Gemma3-12B](https://huggingface.co/google/gemma-3-12b-pt). |
| [MiLMMT-46-12B-v0.1](https://huggingface.co/xiaomi-research/MiLMMT-46-12B-v0.1) | Finetuned on MiLMMT-46-12B-Pretrain with translation instructions. |
| [MiLMMT-46-12B-v1.0](https://huggingface.co/xiaomi-research/MiLMMT-46-12B-v1.0) | Reinforcement learning and model merging on [MiLMMT-46-12B-v0.1](https://huggingface.co/xiaomi-research/MiLMMT-46-12B-v0.1). |


## 🌍 Supported Languages

**GemmaX2-28 (28 languages):** Arabic, Bengali, Czech, German, English, Spanish, Persian, French, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Burmese, Dutch, Polish, Portuguese, Russian, Thai, Tagalog, Turkish, Urdu, Vietnamese, Chinese.

**MiLMMT-46 (46 languages):** Arabic, Azerbaijani, Bulgarian, Bengali, Catalan, Czech, Danish, German, Greek, English, Spanish, Persian, Finnish, French, Hebrew, Hindi, Croatian, Hungarian, Indonesian, Italian, Japanese, Kazakh, Khmer, Korean, Lao, Malay, Burmese, Norwegian, Dutch, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Swedish, Tamil, Thai, Tagalog, Turkish, Urdu, Uzbek, Vietnamese, Cantonese, Chinese (Simplified), Chinese (Traditional).


## 📝 Translation Prompt

The models expect the following prompt format. Use the exact language names listed under [Supported Languages](#-supported-languages).

```text
Translate this from <source language name> to <target language name>:
<source language name>: <source language sentence>
<target language name>:
```


## 🚀 Quick Start

#### vLLM

```python
from vllm import LLM, SamplingParams

model_id = "xiaomi-research/MiLMMT-46-12B-v1.0"

model = LLM(model=model_id)
sampling_params = SamplingParams(top_k=1, temperature=0, max_tokens=2048)

text = "Translate this from Chinese (Simplified) to English:\nChinese (Simplified): 我爱机器翻译\nEnglish:"

outputs = model.generate(text, sampling_params)
print(outputs[0].outputs[0].text)
```

#### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "xiaomi-research/MiLMMT-46-12B-v1.0"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "Translate this from Chinese (Simplified) to English:\nChinese (Simplified): 我爱机器翻译\nEnglish:"
inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


## 🏋️ Training

We train our models with the [LlamaFactory](https://github.com/hiyouga/LlamaFactory) framework. See the [LlamaFactory data docs](https://github.com/hiyouga/LlamaFactory/tree/main/data) for how to register pretraining and finetuning datasets. Remember to add your dataset to `dataset_info.json` before training.

### Continual Pretraining

Data samples for multilingual continual pretraining are in [`examples/cpt.json`](examples/cpt.json). Run:

```bash
bash scripts/cpt.sh
```

### Supervised Finetuning

Data samples for translation instruction finetuning are in [`examples/sft.json`](examples/sft.json). Run:

```bash
bash scripts/sft.sh
```


## 📚 Reference

If you find the resources in this repository helpful, please cite:

```bibtex
@misc{shang2026scalingmodeldatamultilingual,
      title={Scaling Model and Data for Multilingual Machine Translation with Open Large Language Models}, 
      author={Yuzhe Shang and Pengzhi Gao and Wei Liu and Jian Luan and Jinsong Su},
      year={2026},
      eprint={2602.11961},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.11961}, 
}
```

```bibtex
@inproceedings{cui-etal-2025-multilingual,
    title = "Multilingual Machine Translation with Open Large Language Models at Practical Scale: An Empirical Study",
    author = "Cui, Menglong  and
      Gao, Pengzhi  and
      Liu, Wei  and
      Luan, Jian  and
      Wang, Bin",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.280/",
    doi = "10.18653/v1/2025.naacl-long.280",
    pages = "5420--5443",
    ISBN = "979-8-89176-189-6",
    abstract = "Large language models (LLMs) have shown continuously improving multilingual capabilities, and even small-scale open-source models have demonstrated rapid performance enhancement. In this paper, we systematically explore the abilities of open LLMs with less than ten billion parameters to handle multilingual machine translation (MT) tasks. We conduct comprehensive evaluations on six popular LLMs and find that models like Gemma2-9B exhibit impressive multilingual translation capabilities. We then introduce the Parallel-First Monolingual-Second (PFMS) data mixing strategy in the continual pretraining stage to further enhance the MT performance and present GemmaX2-28, a 9B model achieving top-tier multilingual translation performance across 28 languages. Specifically, GemmaX2-28 consistently outperforms the state-of-the-art (SOTA) models such as TowerInstruct and X-ALMA and achieves competitive performance with Google Translate and GPT-4-turbo."
}
```
