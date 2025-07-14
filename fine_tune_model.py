import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load the dataset
train_dataset = load_dataset('json', data_files='data/medical_training_data.json', split='train')

# Label to ID mapping
label_mapping = {label: i for i, label in enumerate(train_dataset.features['label'].names)}

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Initialize the tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))

# Tokenize the dataset
encoded_dataset = train_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()
