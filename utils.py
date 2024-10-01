# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics
import itertools
import numpy as np
import math
import torch
import seaborn as sns
import pandas as pd

MAX = 65535

def step_decay(epoch, initial_lrate=0.01):
    flattern_factor = initial_lrate ** 2.25
    epochs_drop = 5.0
    #drop modelado como modelado no artigo
    drop = initial_lrate **(flattern_factor/epochs_drop)
    
    lrate = initial_lrate * math.pow(drop,  
            math.floor((epoch)/epochs_drop))
    return lrate

def normalize_rgb_ln(X, preprocess=None):
    a = np.log(X)/np.log(65535.0)
    a = a * 255
    if not preprocess:
        return a.astype('uint8')
    return preprocess(a.astype('uint8'))

def plot_confusion_matrix_sns(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    
    #print(cm)
    if normalize:
        #print(cm.sum(axis=1)[:, np.newaxis])
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]+0.0000001
        cm = np.around(cm, decimals=2)
    
    #print(cm)
    df_cm = pd.DataFrame(cm, index = classes, columns=classes)
    
    vmin = np.min(cm)
    vmax = np.max(cm)
    off_diag_mask = np.eye(*cm.shape, dtype=bool)
    fig_size = len(classes) * 1.2
    fig = plt.figure(figsize=(30, 30))

    sns.heatmap(df_cm, annot=True,cmap="OrRd")

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


    return fig
        

#DEFINITION OF TEST METHODS
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure(figsize=(30, 30))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
    

    thresh = cm.max() / (2/3.)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def confusion_matrix(test_data_generator, model, return_fig=False, class_labels=None, steps=None, mode='tensorflow', sns=False, normalize=False):
  
  #tensorflow mode
  #test_data_generator.reset()
  if mode == 'tensorflow':
    if steps == None:
        steps=test_data_generator.samples
    predictions = model.predict(test_data_generator, steps=steps)
    #print(predictions)
    # Get most likely class
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = list(test_data_generator.labels)
  
  #pytorch mode 
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    true_classes = []
    predicted_classes = []
    with torch.no_grad():
        for images, labels in test_data_generator: 
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            true_classes.extend(labels.cpu().numpy().tolist())
            predicted_classes.extend(predicted.cpu().numpy().tolist())
  
  if class_labels == None:
    class_labels = [str(x) for x in np.unique(true_classes)]
  #print(class_labels)  
  #print(len(true_classes))
  report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels, digits=4)
  report_dict = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels, digits=4, output_dict=True)
  cm = metrics.confusion_matrix(true_classes, predicted_classes)
  print(report)
  if not sns:
    fig = plot_confusion_matrix(cm, class_labels, normalize=normalize)
  else:
    fig = plot_confusion_matrix_sns(cm, class_labels, normalize=normalize)
  if return_fig:
      (report, fig)
  return report, report_dict