<div align="center">

# GemmaX: Multilingual Translator based on Gemma Open Models
</div>

<div  align="center">
<img src='/images/gemmax.png' width='600' height='337'>
</div>

## Updates

* Jan. 26 2026: The GemmaX3 paper: [Scaling Model and Data for Multilingual Machine Translation with Open Large Language Models]() is available on ArXiv!

* Jan. 23 2025: The GemmaX2 paper: [Multilingual Machine Translation with Open Large Language Models at Practical Scale: An Empirical Study](https://arxiv.org/abs/2502.02481) has been accepted at **NAACL 2025**!


## Download Models

Model checkpoints are released at huggingface:

#### GemmaX2 Models

| Models                                                                             | Descriptions                                                                                     |
|------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| [GemmaX2-28-2B-Pretrain](https://huggingface.co/ModelSpace/GemmaX2-28-2B-Pretrain) | Developed through continual pretraining of [Gemma2-2B](https://huggingface.co/google/gemma-2-2b) |
| [GemmaX2-28-2B-v0.1](https://huggingface.co/ModelSpace/GemmaX2-28-2B-v0.1)         | Finetuned on GemmaX2-28-2B-Pretrain with translation instructions                                |
| [GemmaX2-28-9B-Pretrain](https://huggingface.co/ModelSpace/GemmaX2-28-9B-Pretrain) | Developed through continual pretraining of [Gemma2-9B](https://huggingface.co/google/gemma-2-9b) |
| [GemmaX2-28-9B-v0.1](https://huggingface.co/ModelSpace/GemmaX2-28-9B-v0.1)         | Finetuned on GemmaX2-28-9B-Pretrain with translation instructions                                |

#### GemmaX3 Models

| Models                      | Descriptions                                                                                          |
|-----------------------------|-------------------------------------------------------------------------------------------------------|
| [GemmaX3-46-1B-Pretrain]()  | Developed through continual pretraining of [Gemma3-1B](https://huggingface.co/google/gemma-3-1b-pt)   |
| [GemmaX3-46-1B-v0.1]()      | Finetuned on GemmaX3-46-1B-Pretrain with translation instructions                                     |
| [GemmaX3-46-4B-Pretrain]()  | Developed through continual pretraining of [Gemma3-4B](https://huggingface.co/google/gemma-3-4b-pt)   |
| [GemmaX3-46-4B-v0.1]()      | Finetuned on GemmaX3-46-4B-Pretrain with translation instructions                                     |
| [GemmaX3-46-12B-Pretrain]() | Developed through continual pretraining of [Gemma3-12B](https://huggingface.co/google/gemma-3-12b-pt) |
| [GemmaX3-46-12B-v0.1]()     | Finetuned on GemmaX3-46-12B-Pretrain with translation instructions                                    |


**Note that GemmaX2-28-2B-Pretrain, GemmaX2-28-9B-Pretrain, GemmaX3-46-1B-Pretrain, GemmaX3-46-4B-Pretrain, and GemmaX3-46-12B-Pretrain are NOT translation models.**

## Supported Languages

GemmaX2 models support 28 languages: Arabic, Bengali, Czech, German, English, Spanish, Persian, French, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Burmese, Dutch, Polish, Portuguese, Russian, Thai, Tagalog, Turkish, Urdu, Vietnamese, Chinese.

GemmaX3 models support 46 languages: Arabic, Azerbaijani, Bulgarian, Bengali, Catalan, Czech, Danish, German, Greek, English, Spanish, Persian, Finnish, French, Hebrew, Hindi, Croatian, Hungarian, Indonesian, Italian, Japanese, Kazakh, Khmer, Korean, Lao, Malay, Burmese, Norwegian, Dutch, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Swedish, Tamil, Thai, Tagalog, Turkish, Urdu, Uzbek, Vietnamese, Cantonese, Chinese (Simplified), Chinese (Traditional).

**Please use the language name specified above in the translation prompt.**

## Quick Start

#### Using on vLLM:
```python3
from vllm import LLM, SamplingParams


model_id = "GemmaX3-46-12B-v0.1"

model = LLM(model=model_id)
sampling_params = SamplingParams(best_of=1, temperature=0, max_tokens=2048)

text = "Translate this from Chinese (Simplified) to English:\nChinese (Simplified): 我爱机器翻译\nEnglish:"

outputs = model.generate(text, sampling_params)
print(outputs[0].outputs[0].text)
```

#### Using on Transformers:
```python3
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "GemmaX3-46-12B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

text = "Translate this from Chinese (Simplified) to English:\nChinese (Simplified): 我爱机器翻译\nEnglish:"
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


The translation prompt is:
```text
Translate this from <source language name> into <target language name>:
<source language name>: <source language sentence>
<target language name>:
```

## Training

We train our models with the [LlamaFactory](https://github.com/hiyouga/LlamaFactory) framework. Please check [here](https://github.com/hiyouga/LlamaFactory/tree/main/data) for adding pretraining and finetuning datasets in `LlamaFactory`. 

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

```

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
