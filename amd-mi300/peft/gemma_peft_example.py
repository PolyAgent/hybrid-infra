import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from peft import get_peft_model
from trl import SFTTrainer

# gemma2 doesn't work yet
# base_model_name = "google/gemma-2-9b"
base_model_name = "google/gemma-2b"

# Model and tokenizer names
new_model_name = "gemma-tuned" #You can give your own name for fine tuned model

# Tokenizer
gemma_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
gemma_tokenizer.pad_token = gemma_tokenizer.eos_token
gemma_tokenizer.padding_side = "right"

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

data_name = "mlabonne/guanaco-llama2-1k"
training_data = load_dataset(data_name, split="train")
# check the data
print(training_data.shape)
# #11 is a QA sample in English
print(training_data[11])

train_params = TrainingArguments(
    output_dir="./test_rsults",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=50,
    learning_rate=4e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb"
)


# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, peft_parameters)
model.print_trainable_parameters()
print('listing model layers')
print(model)

print("==================================================")
print("start SFT")

fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=gemma_tokenizer,
    args=train_params
)

# Training
fine_tuning.train()

