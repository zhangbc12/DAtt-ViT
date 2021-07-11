# -*-coding:utf-8-*-
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap=plt.cm.binary):
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, font1)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label', font1)
    plt.xlabel('Predicted label', font1)

def confusion(y_true, y_pred, labels, name):
    tick_marks = np.array(range(len(labels))) + 0.5
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm_normalized)
    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, labels, title='Normalized confusion matrix')
    # show confusion matrix
    plt.savefig(os.path.join('pic', name + '.png'), format='png')
    plt.show()


if __name__ == '__main__':
    labels = ['AAA', 'BBB', 'CCCC', 'DDD', 'EEE', 'FFF']
    y_true = np.array([0, 0, 1, 2, 1, 2, 0, 2, 2, 0, 1, 1, 4, 3, 3, 5, 5, 4])
    y_pred = np.array([1, 0, 1, 2, 1, 0, 0, 2, 2, 0, 1, 1, 4, 3, 4, 5, 3, 3])
    name = 'test'
    confusion(y_true, y_pred, labels, name)