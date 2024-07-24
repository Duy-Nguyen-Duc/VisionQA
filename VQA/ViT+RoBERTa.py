import torch
import torch.nn as nn
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformer import ViTModel, ViTImageProcessor
from transformer import AutoTokenizer, RobertaModel
import timm

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

classes = set([sample['answer'] for sample in train_data])
classes_to_idx = {
    cls_name: idx for idx, cls_name in enumerate(classes)
}

idx_to_classes = {
    idx: cls_name for idx, cls_name in enumerate(classes)
}

class VQADataset(Dataset):
  def __init__(
      self,
      data,
      classes_to_idx,
      img_feature_extractor,
      text_tokenizer,
      device,
      root_dir='/vqa_coco_dataset/val2014-resised/'
    ):
    self.data = data
    self.classes_to_idx = classes_to_idx
    self.img_feature_extractor = img_feature_extractor
    self.text_tokenizer = text_tokenizer
    self.device = device
    self.root_dir = root_dir
  def __len__(self):
    return len(self.data)
  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir, self.data[index]['image_path'])
    img = Image.open(img_path).convert('RGB')
    if self.img_feature_extractor:
      img = self.img_feature_extractor(img, return_tensors='pt')
      img = {k: v.to(self.device).squeeze(0) for k,v in img.items()}
    question = self.data[index]['question']
    if self.text_tokenizer:
      question = self.text_tokenizer(question, padding = "max_length", max_length = 20, truncation = True, return_tensors='pt')
      question = {k: v.to(self.device).squeeze(0) for k,v in question.items()}
    label = self.data[index]['answer']
    label = torch.tensor(classes_to_idx[label], dtype = torch.long).to(device)
    sample = {
      'image': img,
      'question': question,
      'label': label
    }
    return sample
  
img_feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
text_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

train_dataset = VQADataset(train_data, classes_to_idx, img_feature_extractor, text_tokenizer, device)
test_dataset = VQADataset(test_data, classes_to_idx, img_feature_extractor, text_tokenizer, device)
val_dataset = VQADataset(val_data, classes_to_idx, img_feature_extractor, text_tokenizer, device)

train_batch_size = 128
test_batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

class TextEncoder(nn.Module):
  def __init__(self):
    super(TextEncoder, self).__init__()
    self.model = RobertaModel.from_pretrained('roberta-base')
  def forward(self, inputs):
    outputs = self.model(**inputs)
    return outputs.last_hidden_state[:,0,:]
class VisualEncoder(nn.Module):
  def __init__(self):
    super(VisualEncoder, self).__init__()
    self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
  def forward(self, inputs):
    outputs = self.model(**inputs)
    return outputs.last_hidden_state[:,0,:]
class Classifier(nn.Module):
  def __init__(self, input_dim = 768*2, hidden_dim = 512, n_layers = 1, dropout_prob = 0.2, n_classes =2):
    super(Classifier, self).__init__()
    self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
    self.dropout = nn.Dropout(dropout_prob)
    self.fc1 = nn.Linear(hidden_dim * 2, n_classes)
  def forward(self, inputs):
    x, _ = self.lstm(inputs)
    x = self.dropout(x)
    x = self.fc1(x)
    return x
  
class VQAModel(nn.Module):
  def __init__(self, visual_encoder, text_encoder, classifier):
    super(VQAModel, self).__init__()
    self.visual_encoder = visual_encoder
    self.text_encoder = text_encoder
    self.classifier = classifier
  def forward(self, image, answer):
    text_features = self.text_encoder(answer)
    visual_features = self.visual_encoder(image)
    features = torch.cat((text_features, visual_features), dim=1)
    logits = self.classifier(features)
    return logits
  def freeze(self,visual = True, textual = True, clas = False):
    if visual:
      for n,p in self.visual_encoder.named_parameters():
        p.requires_grad = False
    if textual:
      for n,p in self.text_encoder.named_parameters():
        p.requires_grad = False
    if clas:
      for n,p in self.classifier.named_parameters():
        p.requires_grad = False
      

n_classes = len(classes)
hidden_size = 1024
n_layers = 1
dropout_prob = 0.2

text_encoder = TextEncoder().to(device)
visual_encoder = VisualEncoder().to(device)
classifier = Classifier(hidden_size, hidden_size, n_layers, dropout_prob, n_classes).to(device)
model = VQAModel(visual_encoder, text_encoder, classifier).to(device)
model.freeze()

def evaluate(model, dataloader, criterion, device):
  model.eval()
  correct = 0 
  total = 0
  losses = []
  with torch.no_grad():
    for idx, inputs in enumerate(dataloader):
      image = inputs['image']
      question = inputs['question']
      labels = inputs['label']
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
    for idx, inputs in enumerate(train_dataloader):
      image = inputs['image']
      question = inputs['question']
      labels = inputs['label']
      
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