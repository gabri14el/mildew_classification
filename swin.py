import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
#import mlflow
from torchvision.datasets import ImageFolder

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
from torchvision.models import swin_transformer, resnet50, mobilenet_v3_large, efficientnet_b4, maxvit_t, convnext_tiny,vit_b_16, vit_s_16
import random

from PIL import ImageFilter

from torcheval.metrics import MulticlassF1Score
import mlflow
import utils
import matplotlib.pyplot as plt
import json


num_epochs = 50
dataset_path = '/mnt/c/Users/Gabriel/Downloads/castas-huge2-split'
#dataset_path = '/mnt/c/Users/Gabriel/Downloads/Castas-Dataset-V2-split'
batch_size = 32
input_size = 224
experiment_name = "initial_training"
model_function = vit_s_16
head_name = "heads"
lr = 0.001
wd = 0.0005
comments = "Initial training"

models = [resnet50, swin_transformer.swin_t, mobilenet_v3_large, efficientnet_b4, maxvit_t, convnext_tiny]
tops = ['fc', 'head', 'classifier', 'classifier', 'classifier', 'classifier']


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
def compute_f1_score(predictions, labels):
    true_positives = torch.sum(predictions * labels)
    false_positives = torch.sum(predictions * (1 - labels))
    false_negatives = torch.sum((1 - predictions) * labels)
    
    precision = true_positives / (true_positives + false_positives + 1e-7)
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return f1_score

# Define your dataset and data augmentation transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([transforms.RandomResizedCrop(input_size, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
])

test_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    normalize
])

train_ds = ImageFolder(root=os.path.join(dataset_path, 'train'), transform=train_transform)
val_ds = ImageFolder(root=os.path.join(dataset_path, 'val'), transform=test_transform)
test_ds = ImageFolder(root=os.path.join(dataset_path, 'test'), transform=test_transform)

num_classes = len(train_ds.classes)

new_head = nn.Sequential()

# Define your model
#model = swin_transformer.swin_t(weights="IMAGENET1K_V1")
#n_inputs = model.head.in_features
model = model_function(weights="IMAGENET1K_V1")

if isinstance(model.__dict__['_modules'][head_name], nn.Sequential):
    
    for m in model.__dict__['_modules'][head_name]:
        if not isinstance(m, nn.Linear):
            new_head.append(m)
        else:
            n_inputs = m.in_features
            break
else: 
    n_inputs = model.__dict__['_modules'][head_name].in_features


# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

new_head.append(nn.Dropout(0.2))
new_head.append(nn.Linear(n_inputs, 2048))
new_head.append(nn.GELU())
new_head.append(nn.LayerNorm(2048))
new_head.append(nn.Dropout(0.2))
new_head.append(nn.Linear(2048, num_classes))

model.__dict__['_modules'][head_name] = new_head

for param in model.__dict__['_modules'][head_name].parameters():
    param.requires_grad = True



print(model)
# Define your loss function
criterion = nn.CrossEntropyLoss()

# Define your optimizer and weight decay
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

# Define your learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# Define your data loaders
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

device = torch.device("cuda:1")
model.to(device)
metric = MulticlassF1Score(average="macro", num_classes=num_classes)
print(f"Starting training of {model_function.__name__}...")

mlflow.set_experiment(experiment_name)
run_name = model_function.__name__
path_save =f'{model_function.__name__}_best.pth'
mlflow.start_run(run_name=run_name)
best_f1_score = 0.0
# Training loop
for epoch in range(num_epochs):
    model.train()
    if epoch < 10:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * (epoch + 1) / 10

    train_f1_score = 0.0
    train_loss = 0.0
    metric.reset()
    for images, labels in train_loader:
        images, labels = images.to(device), torch.eye(num_classes)[labels].to(device)
        
        # Perform forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)

        # Perform backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        # Compute training metrics
        predictions = torch.argmax(outputs, dim=1)
        labels = torch.argmax(labels, dim=1)
        metric.update(predictions, labels)
        # Update learning rate
        scheduler.step()
    
    train_f1_score = metric.compute()
    train_loss /= len(train_loader.dataset)
    
    # Validation loop
    model.eval()
    val_f1_score = 0.0
    val_loss = 0.0
    metric.reset()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), torch.eye(num_classes)[labels].to(device)
            # Perform forward pass
            outputs = model(images)

            # Compute validation loss
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            # Compute validation metrics
            predictions = torch.argmax(outputs, dim=1)
            labels = torch.argmax(labels, dim=1)
            metric.update(predictions, labels)
    # Compute average test F1 score
    val_f1_score = metric.compute()
    # Compute average validation loss
    val_loss /= len(val_loader.dataset)

    # Print the test F1 score
    
    log = f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train F1 Score: {train_f1_score:.4f}, Val F1 Score: {val_f1_score:.4f}"
    if val_f1_score > best_f1_score:
        best_f1_score = val_f1_score
        torch.save(model.state_dict(), path_save)
        log+= ", Model saved"
    
    print(log)

#model.load_state_dict(torch.load('model.pth'))

model.load_state_dict(torch.load(path_save,  map_location=torch.device(device)), strict=True)

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = True

lr /= 10

# Define your optimizer and weight decay
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

for epoch in range(num_epochs):
    model.train()
    if epoch < 10:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * (epoch + 1) / 10

    train_f1_score = 0.0
    train_loss = 0.0
    metric.reset()
    for images, labels in train_loader:
        images, labels = images.to(device), torch.eye(num_classes)[labels].to(device)
        
        # Perform forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)

        # Perform backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        # Compute training metrics
        predictions = torch.argmax(outputs, dim=1)
        labels = torch.argmax(labels, dim=1)
        metric.update(predictions, labels)
        # Update learning rate
        scheduler.step()
    
    train_f1_score = metric.compute()
    train_loss /= len(train_loader.dataset)
    
    # Validation loop
    model.eval()
    val_f1_score = 0.0
    val_loss = 0.0
    metric.reset()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), torch.eye(num_classes)[labels].to(device)
            # Perform forward pass
            outputs = model(images)

            # Compute validation loss
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            # Compute validation metrics
            predictions = torch.argmax(outputs, dim=1)
            labels = torch.argmax(labels, dim=1)
            metric.update(predictions, labels)
    # Compute average test F1 score
    val_f1_score = metric.compute()
    # Compute average validation loss
    val_loss /= len(val_loader.dataset)

    # Print the test F1 score
    
    log = f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train F1 Score: {train_f1_score:.4f}, Val F1 Score: {val_f1_score:.4f}"
    if val_f1_score > best_f1_score:
        best_f1_score = val_f1_score
        torch.save(model.state_dict(), path_save)
        log+= ", Model saved"
    
    print(log)


model.load_state_dict(torch.load(path_save,  map_location=torch.device(device)), strict=True)

classes = list(test_ds.class_to_idx.keys())
classes_alias = {'tinto cao': 'TC', 'tinta francisca': 'TF', 'alicante': 'AC', 'alveralhao': 'AV', 'arinto': 'AT', 'bastardo': 'BT', 'boal': 'BA', 'cabernet franc': 'CF', 'cabernet sauvignon': 'CS', 'carignon noir': 'CN', 'cercial': 'CC', 'chardonnay': 'CD', 'codega': 'CG', 'codega do larinho': 'CR', 'cornifesto': 'CT', 'donzelinho': 'DZ', 'donzelinho branco': 'DB', 'donzelinho tinto': 'DT', 'esgana cao': 'EC', 'fernao pires': 'FP', 'folgasao': 'FG', 'gamay': 'GM', 'gouveio': 'GV', 'malvasia corada': 'MC', 'malvasia fina': 'MF', 'malvasia preta': 'MP', 'malvasia rei': 'MR', 'merlot': 'ML', 'moscatel galego': 'MG', 'moscatel galego roxo': 'MX', 'mourisco tinto': 'MT', 'pinot blanc': 'PB', 'rabigato': 'RB', 'rufete': 'RF', 'samarrinho': 'SM', 'sauvignon blanc': 'SB', 'sousao': 'SS', 'tinta amarela': 'TA', 'tinta barroca': 'TB', 'tinta femea': 'TM', 'tinta roriz': 'TR', 'touriga francesa': 'TS', 'touriga nacional': 'TN', 'viosinho': 'VO'}
report, report_dict = utils.confusion_matrix(test_loader, model, class_labels=[classes_alias[c.lower()] for c in classes],mode='pytorch', sns=True, normalize=True)

mlflow.log_param("batch_size", batch_size)
mlflow.log_param("dim", input_size)
mlflow.log_param("dataset", dataset_path)
mlflow.log_param("optimizer", 'AdamW')
mlflow.log_param("lr",lr)
mlflow.log_param("wd",wd)
mlflow.log_artifact(path_save)
mlflow.log_text(report, F"cm.txt")
mlflow.log_text(json.dumps(report_dict), F"json_results.txt")
mlflow.log_figure(plt.gcf(), 'cm.png')
mlflow.log_param("loss", 'cross_entropy')
mlflow.log_param("comments", comments or '')
mlflow.log_metrics(report_dict['macro avg'])
mlflow.log_param("input size", input_size)
mlflow.log_param("model", model_function.__name__)
mlflow.end_run()

torch.cuda.empty_cache()


