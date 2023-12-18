import sys
import transformers
from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

length=128
batch=16
learning_rate=1e-3

# Load model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Load yelp dataseet
dataset = load_dataset("yelp_review_full")

df_train = pd.DataFrame(dataset['train'])
df_test = pd.DataFrame(dataset['test'])

class YelpDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('working with', device)
sys.stdout.flush()

model.to(device)

# Freeze all layers except the last layer
for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

# Create instances of the custom dataset
train_dataset = YelpDataset(texts=df_train['text'].values, labels=df_train['label'].values, tokenizer=tokenizer)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

# Define optimizer and training parameters
num_epochs = 5
# optimizer = AdamW(model.parameters(), lr=2e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

print('max length:', length)
print('batch size:', batch)
print('learning rate:', learning_rate)
print('optimizer:', 'Adam with linear learning rate scheduler')
print('number of epochs:', num_epochs)
print('training data size:', len(df_train))
sys.stdout.flush()

for epoch in range(num_epochs):
    model.train()
    total_steps = len(train_loader)

    train_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += outputs.loss.item()
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs.logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += len(labels)

    accuracy = correct_predictions / total_samples
    avg_train_loss = train_loss / len(train_loader)
    print(f"\rEpoch {epoch + 1}/{num_epochs} [100%] Loss: {avg_train_loss:.4f} Accuracy: {accuracy:.4f}")

# Save the model
model.save_pretrained('yelp_model-128-16-un')