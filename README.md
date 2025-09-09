<div align="center">

# GemmaX: Multilingual Translator based on Gemma Open Models
</div>

<div  align="center">
<img src='/images/gemmax.png' width='450' height='253'>
</div>

GemmaX are many-to-many LLM-based multilingual translation models, which adopt multilingual continual pretraining with Parallel-First Monolingual-Second (PFMS) data mixing strategy and instruction finetuning with high-quality translation prompts.


## Updates

* Jan. 23 2025: The GemmaX2 paper: [Multilingual Machine Translation with Open Large Language Models at Practical Scale: An Empirical Study](https://arxiv.org/abs/2502.02481) has been accepted at **NAACL 2025**!


## Download Models

Model checkpoints are released at huggingface:

| Models                                                                             | Descriptions                                                                                     |
|------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| [GemmaX2-28-2B-Pretrain](https://huggingface.co/ModelSpace/GemmaX2-28-2B-Pretrain) | Developed through continual pretraining of [Gemma2-2B](https://huggingface.co/google/gemma-2-2b) |
| [GemmaX2-28-2B-v0.1](https://huggingface.co/ModelSpace/GemmaX2-28-2B-v0.1)         | Finetuned on GemmaX2-28-2B-Pretrain with translation instructions                                |
| [GemmaX2-28-9B-Pretrain](https://huggingface.co/ModelSpace/GemmaX2-28-9B-Pretrain) | Developed through continual pretraining of [Gemma2-9B](https://huggingface.co/google/gemma-2-9b)                                         |
| [GemmaX2-28-9B-v0.1](https://huggingface.co/ModelSpace/GemmaX2-28-9B-v0.1)         | Finetuned on GemmaX2-28-9B-Pretrain with translation instructions                                |

**Note that GemmaX2-28-2B-Pretrain and GemmaX2-28-9B-Pretrain are NOT translation models.**

## Supported Languages

GemmaX2 models support 28 languages: Arabic, Bengali, Czech, German, English, Spanish, Persian, French, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Burmese, Dutch, Polish, Portuguese, Russian, Thai, Tagalog, Turkish, Urdu, Vietnamese, Chinese.

**Please use the language name specified above in the translation prompt.**

## Quick Start

```python3
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ModelSpace/GemmaX2-28-9B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

text = "Translate this from Chinese to English:\nChinese: 我爱机器翻译\nEnglish:"
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


The translation prompt is:
```text
Translate this from <source language name> into <target language name>:
<source language name>: <source language sentence>
<target language name>:
```

## Training

We train our models with the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework. Please check [here](https://github.com/hiyouga/LLaMA-Factory/tree/main/data) for adding pretraining and finetuning datasets in `LLaMA-Factory`. 

### Continual Pretraining

The data samples for multilingual continual pretraining are listed in `examples/cpt.json`. Check the following command for reference:

```bash
bash scripts/cpt.sh
```

### Supervised Finetuning

The data samples for translation instruction finetuning are listed in `examples/sft.json`. Check the following command for reference:

```bash
bash scripts/sft.sh
```


## Reference
If you find the resources in this repository helpful, please cite as:
```
@misc{cui2025multilingualmachinetranslationopen,
      title={Multilingual Machine Translation with Open Large Language Models at Practical Scale: An Empirical Study}, 
      author={Menglong Cui and Pengzhi Gao and Wei Liu and Jian Luan and Bin Wang},
      year={2025},
      eprint={2502.02481},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.02481}, 
}
```
