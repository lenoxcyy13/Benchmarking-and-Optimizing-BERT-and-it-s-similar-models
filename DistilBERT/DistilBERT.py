# DistilBERT
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
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import DistilBertModel,DistilBertTokenizer
import torch.nn.functional as F



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


def train(model, train_loader, optimizer, criterion, batch_size):
    model.train()
    print('===== Training =====')
    running_loss = 0
    show_acc = 1000
    running_correct = 0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        input_ids, attention_mask, label = data
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        label = label.cuda()

        outputs = model(input_ids, attention_mask)

        outputs = torch.softmax(outputs, dim=1)
        loss = criterion(outputs, label)
        cur_acc = (outputs.argmax(1) == label).sum().item()

        # only show acc after a few batches
        if (i + 1) % show_acc == 0:
            print('acc', cur_acc / (batch_size))
        running_correct += cur_acc
        loss.backward()
        optimizer.step()

        if (i + 1) % show_acc == 0:
            print(f"{i + 1} batches: {i + 1}-batch-avg loss {loss:.6f}")
            print(cur_acc / (batch_size))

        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)  # avg loss on each batch
    epoch_acc = 100. * (running_correct / (len(train_loader.dataset)))
    return epoch_loss, epoch_acc


# val/test a Epoch
def validate(model, test_loader, criterion):
    model.eval()  
    print('===== Validation =====')
    running_loss = 0.0
    running_correct = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            input_ids, attention_mask, label = data
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            label = label.cuda()
            cur_acc = 0

            outputs = model(input_ids, attention_mask)
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
        self.bert = DistilBertModel.from_pretrained(
            r"./distilbert")

        self.MLP = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        for m in self.MLP:
            if type(m) == type(nn.ReLU()):
                continue
            init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, input_ids,attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        o = torch.mean(outputs[0], dim=1)
        text_feat_d = self.drop(o)

        out = F.relu(self.MLP(text_feat_d))
        return out


if __name__ == '__main__':
    class DistilBERT:
        def __init__(self, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
            self.device = torch.device(device)
            self.model = None
            self.optimizer = None
            self.criterion = None
            self.train_loader = None
            self.test_loader = None
            self.train_loss = []
            self.valid_loss = []
            self.train_acc = []
            self.valid_acc = []
            self.save_best_model = None

        def seed_torch(self, seed=4):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        def save_model(self, epochs, save_path='outputs'):
            # save model to 'save_path'
            torch.save(self.model.state_dict(), save_path)

        def save_plots(self, model_name="", optimizer='', save_path='outputs'):
            # plot saving utils
            plt.plot(self.train_loss, label='Train Loss')
            plt.plot(self.valid_loss, label='Valid Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'{model_name} - {optimizer} - Loss')
            plt.legend()
            plt.savefig(f'{save_path}/{model_name}_{optimizer}_loss.png')
            plt.close()

            plt.plot(self.train_acc, label='Train Accuracy')
            plt.plot(self.valid_acc, label='Valid Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title(f'{model_name} - {optimizer} - Accuracy')
            plt.legend()
            plt.savefig(f'{save_path}/{model_name}_{optimizer}_accuracy.png')
            plt.close()

        def train(self, epoch):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (inputs, attention_masks, targets) in enumerate(self.train_loader):
                inputs, attention_masks, targets = inputs.to(self.device), attention_masks.to(self.device), targets.to(
                    self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs, attention_masks)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            epoch_loss = total_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total

            self.train_loss.append(epoch_loss)
            self.train_acc.append(epoch_acc)

            return epoch_loss, epoch_acc

        def validate(self):
            self.model.eval()
            total_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx, (inputs, attention_masks, targets) in enumerate(self.test_loader):
                    inputs, attention_masks, targets = inputs.to(self.device), attention_masks.to(self.device), targets.to(
                        self.device)

                    outputs = self.model(inputs, attention_masks)
                    loss = self.criterion(outputs, targets)

                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            epoch_loss = total_loss / len(self.test_loader)
            epoch_acc = 100. * correct / total

            self.valid_loss.append(epoch_loss)
            self.valid_acc.append(epoch_acc)

            return epoch_loss, epoch_acc

        def train_model(self, epochs, batch_size, lr, dataset_path, tokenizer_path, save_path='outputs'):
            self.seed_torch(1)

            # Load dataset
            dataset = load_dataset(dataset_path)

            # Tokenize
            tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

            train_text = dataset['train']['text']
            train_label = dataset['train']['label']

            train_outputs = tokenizer(train_text, return_tensors="pt", max_length=512,
                                      padding=True, truncation=True)
            train_ids = train_outputs.data['input_ids']
            train_attention_masks = train_outputs.data['attention_mask']
            train_label = torch.tensor(train_label)
            del train_outputs

            test_text = dataset['test']['text']
            test_label = dataset['test']['label']

            test_outputs = tokenizer(test_text, return_tensors="pt", max_length=512,
                                     padding=True, truncation=True)
            test_ids = test_outputs.data['input_ids']
            test_attention_masks = test_outputs.data['attention_mask']
            test_label = torch.tensor(test_label)
            del test_outputs

            self.train_loader = DataLoader(TensorDataset(train_ids, train_attention_masks, train_label),
                                           batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
            self.test_loader = DataLoader(TensorDataset(test_ids, test_attention_masks, test_label),
                                          batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

            self.model = BertMultiLabelCls().to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

            model_name = self.model.__class__.__name__
            opti_name = self.optimizer.__class__.__name__
            save_path = f'{save_path}/{model_name}_{opti_name}_ep{epochs}_lr{lr}'

            self.save_best_model = SaveBestModel(save_path=save_path)

            for epoch in range(epochs):
                print(f"[INFO]: Epoch {epoch + 1} of {epochs}")

                train_epoch_loss, t_acc = self.train(epoch)
                valid_epoch_loss, v_acc = self.validate()

                print(f"Epoch Training loss: {train_epoch_loss:.5f}, training acc: {t_acc:.3f}%")
                print(f"Epoch Validation loss: {valid_epoch_loss:.5f}, validation acc: {v_acc:.3f}%")

                self.save_best_model(valid_epoch_loss, epoch, self.model, self.optimizer, self.criterion)
                print('-' * 50)

            print(f'The best minimal val loss is {self.save_best_model.best_valid_loss}')

            self.save_model(epochs, save_path=save_path)
            self.save_plots(model_name=model_name, optimizer=opti_name, save_path=save_path)


    class SaveBestModel:
        def __init__(self, save_path="outputs", best_valid_loss=float('inf')):
            self.save_path = save_path
            self.best_valid_loss = best_valid_loss

        def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
            if current_valid_loss < self.best_valid_loss:
                self.best_valid_loss = current_valid_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    'best_valid_loss': self.best_valid_loss
                }, self.save_path)


    class BertMultiLabelCls(nn.Module):
        def __init__(self, hidden_size=768, dropout=0.1):
            super(BertMultiLabelCls, self).__init__()
            self.drop = nn.Dropout(dropout)
            self.bert = DistilBertModel.from_pretrained(r"./distilbert")

            self.MLP = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, 5)
            )
            for m in self.MLP:
                if type(m) == type(nn.ReLU()):
                    continue
                init.normal_(m.weight, mean=0, std=0.01)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids, attention_mask)
            o = torch.mean(outputs[0], dim=1)
            text_feat_d = self.drop(o)

            out = F.relu(self.MLP(text_feat_d))
            return out


    if __name__ == '__main__':
        distilbert = DistilBERT()
        distilbert.train_model(epochs=3, batch_size=4, lr=5e-4,
                               dataset_path='./datasets/yelp_review_full',
                               tokenizer_path='./distilbert')
        print('Training completed!')
    print('Training completed!')


