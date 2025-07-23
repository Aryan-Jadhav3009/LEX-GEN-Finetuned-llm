# LexGen – Board Resolution Generator (LoRA Fine-Tuned Mistral)

LexGen is a legal document generation model fine-tuned using LoRA on top of the Mistral-7B base model. It generates structured board resolutions based on input entities and context, streamlining corporate documentation processes.

## 🔧 Model Architecture

- **Base Model**: [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- **Fine-tuning Method**: [LoRA](https://github.com/huggingface/peft)
- **Training Framework**: PEFT + Transformers + Accelerate
- **Tokenizer**: AutoTokenizer from HuggingFace

## 🧠 Training Overview

- **Dataset**: Custom dataset of board resolution templates and field-conditioned examples
- **Epochs**: 200
- **GPU**: T4 (Google Colab)
- **Output**: LoRA adapter weights (`adapter_model.safetensors`) and config (`adapter_config.json`)

## 🚀 Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load base model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load LoRA adapter
adapter_path = "aretoss/LexGen-Mistral-LoRA"
model = PeftModel.from_pretrained(base_model, adapter_path)

# Generate
prompt = "Generate a board resolution for appointing a new director..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(output[0], skip_special_tokens=True))
