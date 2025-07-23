# LexGen – Board Resolution Generator (LoRA Fine-Tuned Mistral)

LexGen is a legal document generation model fine-tuned using LoRA on top of the Mistral-7B base model. It generates structured board resolutions based on input entities and context, streamlining corporate documentation processes.

## 🔧 Model Architecture

- **Base Model**: [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- **Fine-tuning Method**: [LoRA](https://github.com/huggingface/peft)
- **Training Framework**: PEFT + Transformers + Accelerate
- **Tokenizer**: AutoTokenizer from HuggingFace

## 🧠 Training Overview

- **Dataset**: Custom dataset of board resolution templates and field-conditioned examples
- **Training Steps**: 200 (overriding 50 epochs)
- **GPU**: T4 (Google Colab)
- **Output**: LoRA adapter weights (`adapter_model.safetensors`) and config (`adapter_config.json`)

## 🚀 Usage

After cloning this repository and installing dependencies, you can use the adapter with Mistral-7B as follows:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base Mistral model and tokenizer
base_model_name = "mistralai/Mistral-7B-v0.1"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the LoRA adapter from the checkpoint folder
adapter_path = "./checkpoint-200"  # Relative to cloned repo
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# Inference example
instruction = "Generate a board resolution for appointing a new director."
user_input = (
    "Company Name: Aretoss Pvt Ltd\n"
    "Date: 1 June 2025\n"
    "Purpose: Appoint CFO"
)
prompt = f"<s>[INST] {instruction}\n\n{user_input} [/INST]"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
