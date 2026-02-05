<div align="center">

# GemmaX: Multilingual Translator based on Gemma Open Models
</div>

<div  align="center">
<img src='/images/gemmax.png' width='600' height='337'>
</div>

## Updates

* Jan. 23 2025: The GemmaX2 paper: [Multilingual Machine Translation with Open Large Language Models at Practical Scale: An Empirical Study](https://arxiv.org/abs/2502.02481) has been accepted at **NAACL 2025**!


## Download Models

Model checkpoints are released at huggingface:

#### GemmaX2-28 Models

| Models                                                                             | Descriptions                                                                                      |
|------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [GemmaX2-28-2B-Pretrain](https://huggingface.co/ModelSpace/GemmaX2-28-2B-Pretrain) | Developed through continual pretraining of [Gemma2-2B](https://huggingface.co/google/gemma-2-2b). |
| [GemmaX2-28-2B-v0.1](https://huggingface.co/ModelSpace/GemmaX2-28-2B-v0.1)         | Finetuned on GemmaX2-28-2B-Pretrain with translation instructions (v0.1).                         |
| [GemmaX2-28-2B-v0.2](https://huggingface.co/xiaomi-research/GemmaX2-28-2B-v0.2)    | Finetuned on GemmaX2-28-2B-Pretrain with translation instructions (v0.2).                         |
| [GemmaX2-28-9B-Pretrain](https://huggingface.co/ModelSpace/GemmaX2-28-9B-Pretrain) | Developed through continual pretraining of [Gemma2-9B](https://huggingface.co/google/gemma-2-9b). |
| [GemmaX2-28-9B-v0.1](https://huggingface.co/ModelSpace/GemmaX2-28-9B-v0.1)         | Finetuned on GemmaX2-28-9B-Pretrain with translation instructions (v0.1).                         |
| [GemmaX2-28-9B-v0.2](https://huggingface.co/xiaomi-research/GemmaX2-28-9B-v0.2)    | Finetuned on GemmaX2-28-9B-Pretrain with translation instructions (v0.2).                         |


**Note that GemmaX2-28-2B-Pretrain and GemmaX2-28-9B-Pretrain are NOT translation models.**


## Supported Languages

GemmaX2-28 models support 28 languages: Arabic, Bengali, Czech, German, English, Spanish, Persian, French, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Burmese, Dutch, Polish, Portuguese, Russian, Thai, Tagalog, Turkish, Urdu, Vietnamese, Chinese.


## Translation Prompt

```text
Translate this from <source language name> to <target language name>:
<source language name>: <source language sentence>
<target language name>:
```
Please use the language name specified above in the translation prompt.


## Quick Start

#### Using on vLLM:
```python3
from vllm import LLM, SamplingParams


model_id = "xiaomi-research/GemmaX2-28-2B-v0.2"

model = LLM(model=model_id)
sampling_params = SamplingParams(top_k=1, temperature=0, max_tokens=2048)

text = "Translate this from Chinese (Simplified) to English:\nChinese (Simplified): 我爱机器翻译\nEnglish:"

outputs = model.generate(text, sampling_params)
print(outputs[0].outputs[0].text)
```

#### Using on Transformers:
```python3
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "xiaomi-research/GemmaX2-28-2B-v0.2"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "Translate this from Chinese (Simplified) to English:\nChinese (Simplified): 我爱机器翻译\nEnglish:"
inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
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
