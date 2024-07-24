import torch
import torch.nn as nn
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import spacy

def create_dataset(data_path):
  data = []
  with open(data_path, "r") as f:
    lines = f.readlines ()
    for line in lines:
      temp = line.split('\t')
      qa = temp[1].split('?')
      if len(qa) == 3:
        answer = qa[2].strip()
      else:
        answer = qa[1].strip()
      data_sample = {
        'image_path': temp[0][-2], 'question': qa[0] + '?', 'answer': answer
      }
      data.append(data_sample)
  
  return data

train_data = create_dataset('./vqa_coco_dataset/vaq2.0.TrainImages.txt')
val_data = create_dataset('./vqa_coco_dataset/vaq2.0.DevImages.txt')
test_data = create_dataset('./vqa_coco_dataset/vaq2.0.TestImages.txt')


eng = spacy.load('en_core_web_sm')
def get_tokens(data_iter):
  for sample in data_iter:
    question = sample['question']
    yield [token.text for token in eng.tokenizer(question)]


vocab = build_vocab_from_iterator(get_tokens(train_data), min_freq =2, specials=['<pad>','<sos>','<eos>','<unk>'], special_first= True)
vocab.set_default_index(vocab['<unk>'])

classes = set([sample['answer'] for sample in train_data])
classes_to_idx = {
    cls_name: idx for idx, cls_name in enumerate(classes)
}

idx_to_classes = {
    idx: cls_name for idx, cls_name in enumerate(classes)
}

def tokenize(question, max_sequence_length):
  tokens = [token.text for token in eng.tokenizer(question)]
  sequence = [vocab[token] for token in tokens]
  if len(sequence) < max_sequence_length:
    sequence += [vocab['<pad>']] * (max_sequence_length - len(sequence))
  else:
    sequence = sequence[:max_sequence_length]
  return sequence


class VQADataset (Dataset):
  def __init__(
      self,
      data,
      classes_to_idx,
      max_seq_len =30,
      transform=None,
      root_dir='/vqa_coco_dataset/val2014-resised/'):
      self.transform = transform
      self.data = data
      self.max_seq_len = max_seq_len
      self.root_dir = root_dir
      self.classes_to_idx = classes_to_idx
  def __len__(self):
    return len(self.data)
  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir, self.data[index]['image_path'])
    img = Image.open(img_path).convert('RGB')
    if self.transform:
      img = self.transform(img)
    question = self.data[index]['question']
    question = tokenize(question, self.max_seq_len) 
    question = torch.tensor(question, dtype=torch.long)
    label = self.data[index]['answer']
    label = classes_to_idx[label]
    label = torch.tensor(label, dtype=torch.long)
    return img, question, label
  
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset= VQADataset(train_data, classes_to_idx, transform=transform)
val_dataset = VQADataset(val_data, classes_to_idx, transform=transform)
test_dataset = VQADataset(test_data, classes_to_idx, transform=transform)

train_batch_size = 128
test_batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

import timm

class VQAModel(nn.Module):
  def __init__(
      self,
      n_classes,
      img_model_name = 'resnet50',
      embedding_dim = 300,
      n_layers = 2,
      hidden_size = 128,
      dropout_prob = 0.2
    ):
    super(VQAModel, self).__init__()
    self.image_encoder = timm.create_model(img_model_name, pretrained=True, num_classes = hidden_size)
    self.embedding = nn.Embedding(len(vocab), embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, batch_first=True, bidirectional=True)
    self.layernorm = nn.LayerNorm(hidden_size * 2)
    self.fc1 = nn.Linear(hidden_size * 3, 256)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout_prob)
    self.fc2 = nn.Linear(256, n_classes)
  def forward(self, img, text):
    img_features = self.image_encoder(img)
    text_emb = self.embedding(text)
    lstm_out, _ = self.lstm(text_emb)
    lstm_out = lstm_out[:,-1,:]
    lstm_out = self.layernorm(lstm_out)
    combined = torch.cat((img_features, lstm_out), dim=1)
    x = self.fc1(combined)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    return x
  
n_classes = len(classes)
img_model_name = 'resnet50',
embedding_dim = 300,
n_layers = 1
hidden_size = 128,
dropout_prob = 0.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VQAModel(n_classes, img_model_name, embedding_dim, n_layers, hidden_size, dropout_prob).to(device)

def evaluate(model, dataloader, criterion, device):
  model.eval()
  correct = 0 
  total = 0
  losses = []
  with torch.no_grad():
    for image, question, labels in dataloader:
      image = image.to(device)
      question = question.to(device)
      labels = labels.to(device)
      outputs = model(image, question)
      loss = criterion(outputs, labels)
      losses.append(loss.item())
      _, predicted = torch.max(outputs.data,1)
      total += labels.size(0)
      correct += (predicted==labels).sum().item()
  loss = sum(losses) / len(losses)
  accuracy = 100 * correct / total
  return loss, accuracy


def train(model, train_dataloader, val_dataloader, criterion, optimizer,scheduler, device, epochs):
  train_losses = []
  val_losses = []
  for epoch in range(epochs):
    batch_train_losses = []
    model.train()
    for idx, (image, question, labels) in enumerate(train_dataloader):
      image = image.to(device)
      question = question.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = model(image, question)
      loss = criterion(outputs, labels)
      batch_train_losses.append(loss.item())
      loss.backward()
      optimizer.step()
      batch_train_losses.append(loss.item())
    train_loss = sum(batch_train_losses) / len(batch_train_losses)
    train_losses.append(train_loss)
    val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)
    val_losses.append(val_loss)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    scheduler.step()
  return train_losses, val_losses

lr = 1e-2
epochs =10
weight_decay = 1e-5
scheduler_step_size = epochs * 0.6
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

train_losses, val_losses = train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, epochs)