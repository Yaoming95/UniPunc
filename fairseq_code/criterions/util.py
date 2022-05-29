import numpy as np
import pandas as pd


def evaluate(y_pred, y_test, labels, label_names):
    from sklearn import metrics
    # no_used_base = [0, 1, 2, 3]
    no_used_base = []
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=no_used_base+labels)
    overall = metrics.precision_recall_fscore_support(
        y_test, y_pred, average='micro', labels=no_used_base+labels)
    result = pd.DataFrame(
        np.array([precision[len(no_used_base):], recall[len(no_used_base):], f1[len(no_used_base):]]),
        columns=label_names,
        index=['Precision', 'Recall', 'F1']
    )
    result['OVERALL'] = overall[:3]
    return result