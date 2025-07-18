#!/usr/bin/env python3
"""
Open-Source Medical LLM Fine-Tuning Script
==========================================

This script fine-tunes open-source language models on medical data using
the new dataset format (with 'input' and 'intent' fields).

Usage:
    python fine_tune_opensource.py --model microsoft/DialoGPT-medium --data data/medical_training_data.json
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add backend to path
sys.path.append('backend')

from backend.opensource_medical_llm import OpenSourceMedicalLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_medical_data(data_path: str) -> List[Dict[str, Any]]:
    """Load medical training data from JSON file"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} medical training examples")
        return data
    
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []

def prepare_training_data(raw_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Prepare data for training with the new format"""
    prepared_data = []
    
    for item in raw_data:
        # Extract fields from new format
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        intent = item.get('intent', 'unknown')
        category = item.get('category', 'general')
        risk_level = item.get('risk_level', 'low')
        
        if input_text and output_text:
            # Format as training example
            training_example = {
                'input': input_text,
                'output': output_text,
                'intent': intent,
                'category': category,
                'risk_level': risk_level
            }
            prepared_data.append(training_example)
    
    logger.info(f"Prepared {len(prepared_data)} training examples")
    return prepared_data

def analyze_dataset(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the dataset and provide statistics"""
    if not data:
        return {}
    
    # Count by category
    categories = {}
    intents = {}
    risk_levels = {}
    
    for item in data:
        category = item.get('category', 'unknown')
        intent = item.get('intent', 'unknown')
        risk_level = item.get('risk_level', 'unknown')
        
        categories[category] = categories.get(category, 0) + 1
        intents[intent] = intents.get(intent, 0) + 1
        risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
    
    analysis = {
        'total_examples': len(data),
        'categories': categories,
        'intents': intents,
        'risk_levels': risk_levels,
        'average_input_length': sum(len(item.get('input', '')) for item in data) / len(data),
        'average_output_length': sum(len(item.get('output', '')) for item in data) / len(data)
    }
    
    return analysis

def main():
    """Main fine-tuning function"""
    parser = argparse.ArgumentParser(description='Fine-tune open-source medical LLM')
    parser.add_argument('--model', 
                       default='microsoft/DialoGPT-medium',
                       help='Model name to fine-tune')
    parser.add_argument('--data', 
                       default='data/medical_training_data.json',
                       help='Path to medical training data')
    parser.add_argument('--output', 
                       default='./fine_tuned_medical_model',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--epochs', 
                       type=int, 
                       default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', 
                       type=int, 
                       default=2,
                       help='Training batch size')
    parser.add_argument('--learning-rate', 
                       type=float, 
                       default=5e-5,
                       help='Learning rate')
    parser.add_argument('--analyze-only', 
                       action='store_true',
                       help='Only analyze dataset, don\'t train')
    
    args = parser.parse_args()
    
    # Load medical data
    logger.info(f"Loading medical data from {args.data}")
    raw_data = load_medical_data(args.data)
    
    if not raw_data:
        logger.error("No training data available. Exiting.")
        return 1
    
    # Analyze dataset
    analysis = analyze_dataset(raw_data)
    logger.info("Dataset Analysis:")
    logger.info(f"  Total examples: {analysis['total_examples']}")
    logger.info(f"  Categories: {analysis['categories']}")
    logger.info(f"  Intents: {analysis['intents']}")
    logger.info(f"  Risk levels: {analysis['risk_levels']}")
    logger.info(f"  Average input length: {analysis['average_input_length']:.1f} chars")
    logger.info(f"  Average output length: {analysis['average_output_length']:.1f} chars")
    
    if args.analyze_only:
        logger.info("Analysis complete. Exiting.")
        return 0
    
    # Prepare training data
    logger.info("Preparing training data...")
    prepared_data = prepare_training_data(raw_data)
    
    if not prepared_data:
        logger.error("No valid training examples found. Exiting.")
        return 1
    
    # Initialize model
    logger.info(f"Initializing model: {args.model}")
    try:
        medical_llm = OpenSourceMedicalLLM(
            model_name=args.model,
            use_quantization=True
        )
        
        # Check if model is available
        model_info = medical_llm.get_model_info()
        if not model_info['is_loaded']:
            logger.error("Model failed to load. Exiting.")
            return 1
        
        logger.info(f"Model loaded successfully: {model_info}")
        
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return 1
    
    # Start fine-tuning
    logger.info("Starting fine-tuning process...")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Training examples: {len(prepared_data)}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Output directory: {args.output}")
    
    try:
        # Fine-tune the model
        success = medical_llm.fine_tune_model(
            training_data=prepared_data,
            output_dir=args.output
        )
        
        if success:
            logger.info("Fine-tuning completed successfully!")
            logger.info(f"Model saved to: {args.output}")
            
            # Test the fine-tuned model
            logger.info("Testing fine-tuned model...")
            test_queries = [
                "What is a headache?",
                "I have chest pain",
                "How do I prevent diabetes?",
                "What are the symptoms of fever?"
            ]
            
            for query in test_queries:
                try:
                    response = medical_llm.process_query(query)
                    logger.info(f"Test Query: {query}")
                    logger.info(f"Response: {response.response[:100]}...")
                    logger.info(f"Risk Level: {response.risk_level}")
                    logger.info("---")
                except Exception as e:
                    logger.error(f"Error testing query '{query}': {e}")
            
            return 0
            
        else:
            logger.error("Fine-tuning failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
