# RoBERTa model
import glob
import random
import os

import timm
from PIL import Image
import numpy as np
import torch
from torch.nn import init

from datasets import load_dataset
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
from transformers import RobertaTokenizer,RobertaModel


def seed_torch(seed=4):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SaveBestModel:
    def __init__(
            self, save_path="outputs", best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            print(f"create {save_path}/ dir")

    def __call__(
            self, current_valid_loss,
            epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, f'{self.save_path}/best_model.pth')


def save_model(epochs, model, optimizer, criterion, save_path='outputs'):
    print(f"Saving final model...")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, f'{save_path}/final_model.pth')


plt.style.use('ggplot')


def save_plots(train_acc, valid_acc, train_loss, valid_loss, model_name="", optimizer='', save_path='outputs'):
    # loss
    fig = plt.figure(figsize=(7, 5))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )

    fig.suptitle('model:' + model_name + " " + optimizer, fontsize=20)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_path}/loss.png')

    # accuracy
    fig = plt.figure(figsize=(7, 5))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )

    fig.suptitle('model:' + model_name + " " + optimizer, fontsize=20)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{save_path}/accuracy.png')


def train(model, train_loader, optimizer, criterion):
    model.train()
    print('===== Training =====')
    running_loss = 0
    show_acc = 100
    running_correct = 0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        input_ids, attention_mask, label = data
        input_ids = input_ids.cuda()
        # token_type_ids = token_type_ids.cuda()
        attention_mask = attention_mask.cuda()
        label = label.cuda()

        outputs = model(input_ids, attention_mask)

        # forward pass
        outputs = torch.softmax(outputs, dim=1)
        # calculate the loss
        loss = criterion(outputs, label)
        # acc
        cur_acc = (outputs.argmax(1) == label).sum().item()
        if (i + 1) % show_acc == 0:
            print('acc', cur_acc / (batch_size))
        running_correct += cur_acc
        loss.backward()
        optimizer.step()

        if (i + 1) % show_acc == 0:
            print(f"{i + 1} batches: {i + 1}-batch-avg loss {loss:.6f}")
            print(cur_acc / (batch_size))

        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)  
    epoch_acc = 100. * (running_correct / (len(train_loader.dataset)))
    return epoch_loss, epoch_acc


# val/test a Epoch
def validate(model, test_loader, criterion):
    model.eval()  # close BN and dropout layer
    print('===== Validation =====')
    running_loss = 0.0
    running_correct = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            input_ids, attention_mask, label = data
            input_ids = input_ids.cuda()
            # token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()
            label = label.cuda()
            cur_acc = 0

            # forward pass
            outputs = model(input_ids, attention_mask)

            # calculate the loss
            outputs = torch.softmax(outputs, dim=1)

            loss = criterion(outputs, label)
            cur_acc += (outputs.argmax(1) == label).sum().item()
            running_correct += cur_acc

            running_loss += loss.item()

    epoch_loss = running_loss / len(test_loader)  
    epoch_acc = 100. * (running_correct / (len(test_loader.dataset)))

    return epoch_loss, epoch_acc


class BertMultiLabelCls(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.1):
        super(BertMultiLabelCls, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.bert = RobertaModel.from_pretrained(
            r"model/roberta-base")

        self.MLP = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        for m in self.MLP:
            if type(m) == type(nn.ReLU()):
                continue
            init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, input_ids ,attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        text_feat_d = self.drop(outputs[1])

        out = F.relu(self.MLP(text_feat_d))
        return out

class RoBERTaTrainer:
    def __init__(self, epochs=300, batch_size=4, lr=5e-6):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset = load_dataset('./datasets/yelp_review_full')
        self.vocab_file = './model/vocab.json'
        self.merges_file = './model/merges.txt'
        self.tokenizer = RobertaTokenizer(self.vocab_file, self.merges_file)
        self.train_text = self.dataset['train']['text']
        self.train_label = self.dataset['train']['label']
        self.test_text = self.dataset['test']['text']
        self.test_label = self.dataset['test']['label']
        self.model = BertMultiLabelCls()
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.model_name = self.model.__class__.__name__
        self.opti_name = self.optimizer.__class__.__name__
        self.save_path = self.model_name + "_" + self.opti_name + "_ep" + str(self.epochs) + "_lr" + str(self.lr)
        self.save_best_model = SaveBestModel(save_path=self.save_path)
        self.train_loss, self.valid_loss = [], []
        self.train_acc, self.valid_acc = [], []

    def preprocess_data(self):
        train_outputs = self.tokenizer(self.train_text, return_tensors="pt", max_length=512,
                                       padding=True, truncation=True)
        self.train_ids = train_outputs.data['input_ids']
        self.train_attention_masks = train_outputs.data['attention_mask']
        self.train_label = torch.tensor(self.train_label)
        del train_outputs

        test_outputs = self.tokenizer(self.test_text, return_tensors="pt", max_length=512,
                                      padding=True, truncation=True)
        self.test_ids = test_outputs.data['input_ids']
        self.test_attention_masks = test_outputs.data['attention_mask']
        self.test_label = torch.tensor(self.test_label)
        del test_outputs

        train_loader = TensorDataset(self.train_ids, self.train_attention_masks, self.train_label)
        test_loader = TensorDataset(self.test_ids, self.test_attention_masks, self.test_label)

        self.train_loader = DataLoader(train_loader, self.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.test_loader = DataLoader(test_loader, self.batch_size, shuffle=False, num_workers=0, drop_last=False)

    def train_model(self):
        for epoch in range(self.epochs):
            print(f"[INFO]: Epoch {epoch + 1} of {self.epochs}")

            train_epoch_loss, t_acc = train(self.model, self.train_loader, self.optimizer, self.criterion)
            valid_epoch_loss, v_acc = validate(self.model, self.test_loader, self.criterion)

            self.train_loss.append(train_epoch_loss)
            self.valid_loss.append(valid_epoch_loss)
            self.train_acc.append(t_acc)
            self.valid_acc.append(v_acc)
            print(f"Epoch Training loss: {train_epoch_loss:.5f}, training acc: {t_acc:.3f}%")
            print(f"Epoch Validation loss: {valid_epoch_loss:.5f}, validation acc: {v_acc:.3f}%")
            self.save_best_model(valid_epoch_loss, epoch, self.model, self.optimizer, self.criterion)
            print('-' * 50)

        print(f'The best minimal val loss is {self.save_best_model.best_valid_loss}')

        save_model(self.epochs, self.model, self.optimizer, self.criterion, save_path=self.save_path)
        save_plots(self.train_acc, self.valid_acc, self.train_loss, self.valid_loss, model_name=self.model_name,
                   optimizer=self.opti_name, save_path=self.save_path)
        print('Training completed')

    def run(self):
        seed_torch(1)
        self.preprocess_data()
        self.train_model()


if __name__ == '__main__':
    trainer = RoBERTaTrainer()
    trainer.run()
