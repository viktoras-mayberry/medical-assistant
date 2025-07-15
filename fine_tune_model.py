import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# Disable telemetry to avoid connection errors
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "0"

def load_medical_data(file_path):
    """Load medical training data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['input'] for item in data]
    labels = [item['intent'] for item in data]
    return texts, labels

def create_dataset(texts, labels, tokenizer, max_length=512):
    """Create a dataset with tokenized inputs and labels"""
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    return dataset

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def train_medical_model():
    """Main training function"""
    try:
        # Load the medical data
        texts, labels = load_medical_data('data/medical_training_data.json')
        print(f"Loaded {len(texts)} training examples")
        
        # Create label mappings
        unique_labels = list(set(labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        id_to_label = {i: label for label, i in label_to_id.items()}
        print(f"Found {len(unique_labels)} unique intents: {unique_labels}")
        
        # Convert labels to numeric IDs
        label_ids = [label_to_id[label] for label in labels]
        
        # Initialize tokenizer and model
        model_name = "distilbert-base-uncased"
        print(f"Loading tokenizer and model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(unique_labels))
        
        # Create dataset
        dataset = create_dataset(texts, label_ids, tokenizer)
        
        # Split dataset into train/eval
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
        print(f"Training set size: {train_size}, Evaluation set size: {eval_size}")
        
        # Create output directories
        os.makedirs('./results', exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
        os.makedirs('./fine_tuned_medical_model', exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        
        print("Starting training...")
        train_result = trainer.train()
        print(f"Training completed successfully! Train loss: {train_result.training_loss}")
        
        # Save the model
        print("Saving model...")
        try:
            trainer.save_model('./fine_tuned_medical_model')
            print("Model saved successfully!")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise
            
        try:
            tokenizer.save_pretrained('./fine_tuned_medical_model')
            print("Tokenizer saved successfully!")
        except Exception as e:
            print(f"Error saving tokenizer: {e}")
            raise
        
        # Save label mappings
        try:
            with open('./fine_tuned_medical_model/label_mappings.json', 'w') as f:
                json.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f, indent=2)
            print("Label mappings saved successfully!")
        except Exception as e:
            print(f"Error saving label mappings: {e}")
            raise
        
        print("Training completed!")
        print(f"Model saved to: ./fine_tuned_medical_model")
        print(f"Number of labels: {len(unique_labels)}")
        print(f"Labels: {unique_labels}")
        
        # Verify the saved files
        print("\nVerifying saved files:")
        saved_files = os.listdir('./fine_tuned_medical_model')
        print(f"Files in fine_tuned_medical_model: {saved_files}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    train_medical_model()
