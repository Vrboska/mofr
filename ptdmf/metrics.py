""" Basic prediction models' evaluation metrics.

We start with binary model prediction metrics.
For these, we always assume that the scores and labels
are possitively correlated i.e. the higher scores correspond
to observations with more density of 1 labels and the lower scores 
correspond to observations with relatively more 0's than 1's."""

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc


from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve




# scalar binary classification metrics

def gini(y_true, y_score, *args, **kwargs):
    """GINI coefficient in this case is simply scaled
    roc_auc_score in a way that the result is between -1 and 1."""

    return 2*roc_auc_score(y_true, y_score, *args, **kwargs)-1

def liftN(y_true, y_score, p):
    """ p percent lift e.g. p=0.1 calculates 10% lift.
    In this case 10% highest scores would be considered."""
    
    if p==1.0:
        return 1

    q=1-p
    quantile=np.quantile(y_score, q)
    bad_rate_quant=sum(y_true[y_score>q])/len(y_true[y_score>q])
    bad_rate_overall=sum(y_true)/len(y_true)

    return bad_rate_quant/bad_rate_overall

def lift(y_true, y_score):
    """By default we calculate 10% lift."""

    return liftN(y_true, y_score, 0.1)


def ks_score(data1, data2, alternative='two-sided', mode='auto'):
    """Kolmogorov smirnov statistic."""

    return ks_2samp(data1, data2, alternative='two-sided', mode='auto').statistic





