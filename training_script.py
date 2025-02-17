import os
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import PeftModelForCausalLM, LoraModel
import torch

# Load dataset
dataset_path = 'datasets/wikitext_subset.csv'
dataset = pd.read_csv(dataset_path)

# Extract the text data
text_data = dataset['text'].tolist()  # Assuming your CSV has a column 'text'

# Convert text data into Hugging Face Dataset format
train_dataset = Dataset.from_dict({'text': text_data})

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)

# Define LoRA model (you can load the base model here if you want to use LoRA)
lora_model = PeftModelForCausalLM.from_pretrained(model, LoraModel)
lora_model = lora_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",             # Output directory
    evaluation_strategy="epoch",        # Evaluation strategy to use
    per_device_train_batch_size=4,      # Batch size for training
    per_device_eval_batch_size=4,       # Batch size for evaluation
    num_train_epochs=1,                 # Number of training epochs
    logging_dir="./logs",               # Directory for storing logs
    logging_steps=10,
)

# Trainer setup
trainer = Trainer(
    model=lora_model,                       # The model to be trained
    args=training_args,                     # Training arguments
    train_dataset=train_dataset,            # Training dataset
    eval_dataset=None,                      # No evaluation dataset in this case
)

# Start training
trainer.train()

# Save model (this step may be skipped as you indicated you don't want to save the model)
# lora_model.save_pretrained('./model')
