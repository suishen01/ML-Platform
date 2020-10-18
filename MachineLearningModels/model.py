import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

class Model:
    def __init__(self):
        self.type = None
        self.mapping_dict = None
        self.label_headers = None

    def load(self, path, type):
        self.model = pickle.load(open(path, 'rb'))
        self.type = type
        return self.model

    def plot_confusion_matrix(self, y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='Predicted label',
               xlabel='True label')

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

        plt.show()
        return ax

    def map_str_to_number(self, Y):
        mapping_flag = False
        Y_new = Y.copy()
        if self.mapping_dict is not None:
            for label_header in self.label_headers:
                Y_new[label_header] = Y_new[label_header].map(self.mapping_dict)
            return Y_new

        mapping_dict = None
        for label_header in self.label_headers:
            check_list = pd.Series(Y_new[label_header])
            for item in check_list:
                if type(item) == str:
                    mapping_flag = True
                    break
            if mapping_flag:
                classes = Y_new[label_header].unique()
                mapping_dict = {}
                index = 0
                for c in classes:
                    mapping_dict[c] = index
                    index += 1

                Y_new[label_header] = Y_new[label_header].map(mapping_dict)
                mapping_flag = False

        self.mapping_dict = mapping_dict
        return Y_new

    def map_number_to_str(self, Y, classes):
        if self.mapping_dict is not None:
            mapping_dict = self.mapping_dict
        else:
            Y = Y.round()
            mapping_dict = {}
            index = 0
            for c in classes:
                mapping_dict[index] = c
                index += 1
        return Y.map(mapping_dict)

    def getROC(self, test_labels, predictions, label_headers):
        predictions=pd.DataFrame(data=predictions)
        predictions.columns=test_labels.columns.values
        if self.type == 'classifier':
            test_labels = self.map_str_to_number(test_labels)
            predictions = self.map_str_to_number(predictions)
            fpr, tpr, _ = roc_curve(test_labels, predictions)
            plt.figure(1)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr, tpr)
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.show()
        else:
            return 'No Confusion Matrix for Regression'
