import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Load the dataset
df = pd.read_csv('sample_data.csv')

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], examples['question'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
