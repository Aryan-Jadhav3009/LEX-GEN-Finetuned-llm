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

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 1) Load base model & tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2) Load your LoRA adapter
# If you’ve pushed to Hugging Face hub under aretoss/LexGen-Mistral-LoRA:
# adapter_path = "aretoss/LexGen-Mistral-LoRA"
#
# Or if you’re loading locally from checkpoint-200:
# adapter_path = "/path/to/checkpoint-200"
adapter_path = "aretoss/LexGen-Mistral-LoRA"

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# 3) Generate a board resolution
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
print(tokenizer.decode(outputs[0][ inputs.input_ids.shape[-1] : ], skip_special_tokens=True))
