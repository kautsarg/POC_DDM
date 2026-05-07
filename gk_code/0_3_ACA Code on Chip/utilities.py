import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
sns.set_style("ticks")

# Set suitable default colours for colorblind
import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import to_hex
wong_palette_10 = [
    (0, 0.45, 0.70),    # Blue
    (0.9, 0.6, 0.0),    # Orange
    (0.0, 0.60, 0.50),  # Bluish Green
    (0.8, 0.4, 0),      # Vermillion
    (0.35, 0.7, 0.9),   # Sky Blue
    (0.8, 0.6, 0.7),    # Reddish Purple
    (0.5, 0.5, 0.5),    # Gray
    (0.95, 0.9, 0.25),  # Yellow
    (0.5, 0.2, 0.5),    # Deep Purple (New)
    (0.6, 0.1, 0.1)     # Dark Red (New)
]

mpl.rcParams['axes.prop_cycle'] = cycler(color=[to_hex(i) for i in wong_palette_10])

#############################
# Import utility packages
#############################
from tqdm.auto import tqdm
from pathlib import Path
import dill as pickle
from collections import OrderedDict
from functools import reduce

#############################
# Set constants
#############################
DPI = 100 # This is the quality of figures



def unnormalized(df):
    df_unscaled = df.copy()

    time_cols = [col for col in df.columns if col.startswith("Cycle")]
    df_unscaled[time_cols] = df[time_cols].multiply(df['row_max'] - df['row_min'], axis=0).add(df['row_min'], axis=0)
    
    return df_unscaled

def interpolate_melting(df):
    # Interpolate between temperatures
    return df.iloc[:, [0]].join(df.iloc[:, 1:].transpose().interpolate().dropna().transpose())

def plot_curves(ax, data, target, name, color, data_type, xlabel, ylabel, title):
    df_temp = data.loc[data.well==target, :]
    if data_type == 'AC':
        df_temp = df_temp.filter(like='Cycle').transpose()
        df_temp = df_temp.reset_index(drop=True)
    elif data_type == 'MC':
        df_temp = df_temp.iloc[:, 1:].transpose()
        df_temp.index = df_temp.index.astype(float)
    
    ax.plot( df_temp.index, df_temp.values , color=color )
    ax.set_title(f'{title}: {name} (n={df_temp.shape[1]})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix2(y_true, y_pred, classes, mask, ax,
                          normalize=False, labels=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap='Greens')
    
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    accuracy = 100 * np.mean(y_true==y_pred)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix\n(Acc: {:.2f}%)'.format(accuracy),
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.grid(False)
    
    return cm


def my_plot_confusion_matrix2(y_true, y_pred, classes, mask, ax,
                              normalize=False,
                              title=True, test_classes=None,
                              cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if test_classes is None:
        test_classes = classes
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(test_classes))))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cm = cm[:len(test_classes), :len(classes)]
        
    im = ax.imshow(mask, interpolation='nearest', cmap='Greens')
    
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    accuracy = 100 * np.mean(y_true==y_pred)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=test_classes,
           ylabel='True label',
           xlabel='Predicted label')

    if title:
        ax.set_title('Confusion Matrix (Acc: {:.2f}%)'.format(accuracy))
    else:
        ax.set_title('Confusion Matrix')
    
    # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = 302
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black")

    ax.grid(False)
    
    return cm

# def plot_(features, ax, true_each_fold, pred_each_fold, names):
#     output_individual = {}
#     output_individual['y_true'] = np.concatenate([item[features] 
#                                                   for item in true_each_fold])
#     output_individual['y_pred'] = np.concatenate([item[features] 
#                                                   for item in pred_each_fold])
    
    
#     cm = plot_confusion_matrix2(y_true=output_individual['y_true'], 
#                             y_pred=output_individual['y_pred'], 
#                             classes=[names[i] for i in features],
#                             mask=np.eye(len(features), len(features)),
#                             ax=ax)
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     plt.xticks(rotation=0)
    
#     if len(features) == 2:
#         TP = cm[0, 0]
#         FP = cm[0, 1]
#         FN = cm[1, 0]
#         TN = cm[1, 1]


#     #     lab = '{}' + ' vs {}'*(len(features)-1)
#     #     print(lab.format(*[names[i] for i in features]))
#         ax.set_title(ax.get_title()+f'\n(Sens: {100*TP/(TP+FN):.2f}%)\n(Spec: {100*TN/(TN+FP):.2f}%)')
#     #     print(f'Sensitivity: {TP/(TP+FN)}')
#     #     print(f'Specificity: {TN/(TN+FP)}')
#     #     print('Accuracy: {}'.format((output_individual['y_true']==output_individual['y_pred']).mean()))
    

def plot_(features, ax, true_each_fold, pred_each_fold, names):
    # Filter out empty arrays from folds that might have lacked specific class pairs
    y_true = np.concatenate([item[features] for item in true_each_fold if len(item[features]) > 0])
    y_pred = np.concatenate([item[features] for item in pred_each_fold if len(item[features]) > 0])
    
    # Define labels based on the specific features being compared
    current_labels = list(features)
    current_names = [names[i] for i in features]
    
    cm = plot_confusion_matrix2(
        y_true=y_true, 
        y_pred=y_pred, 
        classes=current_names,
        labels=current_labels, # Explicitly tell CM which labels to look for
        mask=np.eye(len(features)),
        ax=ax
    )
    
    # Binary specific metrics (Sensitivity/Specificity)
    if len(features) == 2:
        # Avoid division by zero
        tp, fp, fn, tn = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        sens = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
        ax.set_title(ax.get_title() + f'\nSens: {sens:.1f}% | Spec: {spec:.1f}%', fontsize=9)
    
    ax.set_xlabel('')
    ax.set_ylabel('')