# Import libraries
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Load and preprocess data
# Assuming you have a dataset with 'query' and 'intent' columns
data = pd.read_csv('english_intent_data.csv') # Replace with your dataset
X = data['query']
y = data['intent']

# Vectorization
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=4)

# Create dataset class
class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, truncation=True, padding='max_length', return_tensors="pt")
        return {key: val.squeeze() for key, val in encoding.items()}, torch.tensor(label)

# Prepare dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(data['query'], data['intent'], test_size=0.2)
train_dataset = IntentDataset(train_texts, train_labels)
test_dataset = IntentDataset(test_texts, test_labels)

# Training arguments
training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=16, warmup_steps=500, weight_decay=0.01, logging_dir='./logs', logging_steps=10)

# Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)

# Train and evaluate
trainer.train()
trainer.evaluate()

from transformers import BertTokenizer, BertForSequenceClassification

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=4)

# Use the same dataset class and training code as for RoBERTa

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# Load tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=4)

# Use the same dataset class and training code as for RoBERTa, but with Hindi data

# You will need to implement a custom feature extractor for BIO tagging
# For simplicity, here's a placeholder
# Implement the actual BIO tagging model and feature extraction as needed

# Load and preprocess data
data = pd.read_csv('english_entity_data.csv') # Replace with your dataset
X = data['query']
y = data['BIO_tags']

# Vectorization
# You may need a different feature extraction method for BIO tagging
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# This follows the same pattern as Intent Recognition using RoBERTa, but with BIO tagging

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=num_BIO_tags)

# Use the same dataset class and training code as for intent recognition but modify the labels to BIO tags

from transformers import BertForTokenClassification

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = BertForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=num_BIO_tags)

# Use the same dataset class and training code as for intent recognition but modify the labels to BIO tags

from transformers import XLMRobertaForTokenClassification

# Load tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=num_BIO_tags)

# Use the same dataset class and training code as for intent recognition but modify the labels to BIO tags
