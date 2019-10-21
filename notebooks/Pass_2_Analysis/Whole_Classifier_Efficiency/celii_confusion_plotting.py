# confusion matrix function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix_from_raw_data(y_true, y_pred, classes,cm_labels=[],
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Example: 
    y_true = [0, 0, 1, 1, 2, 0, 1]
    y_pred = [0, 1, 0, 1, 2, 2, 1]
    
    classes = np.array(["Apical","Basal","Oblique"])
    plot_confusion_matrix_from_raw_data(y_true,y_pred,normalize=True,classes=classes,
                         title="Brendan's Confusion")
    done
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    if len(cm_labels) <= 0:
        cm = confusion_matrix(y_true, y_pred)
    else:
        cm = confusion_matrix(y_true, y_pred,labels=cm_labels)
    # Only use the labels that appear in the data
    #print(list(np.unique(np.hstack((y_true,y_pred)))))
    #print(classes)
    
    
    #**** don't need to do any editing to the classes **** #
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    fig.set_size_inches(20, 12)
    ax.grid(False)
    return ax

def plot_confusion_matrix_from_confusion_matrix(cm, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    This version takes in a confusion matrix
    
    Example: 
    Example: 
    y_true = [0, 0, 1, 1, 2, 0, 1]
    y_pred = [0, 1, 0, 1, 2, 2, 1]
    C = confusion_matrix(y_true, y_pred)
    
    classes = np.array(["Apical","Basal","Oblique"])
    plot_confusion_matrix_from_confusion_matrix(C,normalize=True,classes=classes,
                     title="Brendan's Confusion")
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    #**** don't need to do any editing to the classes **** #
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = np.nanmax(cm) / 2.
    #print("threshold = " + str(thresh))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            #print("cm[i,j] = " + str(cm[i,j]))
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                   fontsize=24)
    fig.tight_layout()
    
    fig.set_size_inches(20, 12)
    ax.grid(False)
    return ax,fig