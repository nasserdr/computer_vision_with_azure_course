import numpy as np
from scipy.sparse import coo_matrix
from collections import OrderedDict as Dict

def my_confusion_matrix(y_true, y_pred, n_targets):
    """
    Adapted from sklearn.metrics.confusion_matrix,
    simplified for speed up.
    The confusion matrix is not normalized.
    Args:
        - y_true : 1-D ndarray of integers correponding to the ground truth 
        - y_pred : 1-D ndarray of integers containing the predictions
        - n_targets : number of classes. 
    Output:
        - cm (ndarray) : confusion matrix
    In y_true and y_pred, only integers n with 0 <= n < n_targets are considered. All other values
    are ignored.
    """

    # intersect y_pred, y_true with targets, eliminate items not in targets
    ind = np.logical_and(y_pred < n_targets, y_true < n_targets)
    y_pred = y_pred[ind]
    y_true = y_true[ind]

    sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    cm = coo_matrix((sample_weight, (y_true, y_pred)),
                    shape=(n_targets, n_targets), dtype=np.int64,
                    ).toarray()
    with np.errstate(all='ignore'):
        cm = np.nan_to_num(cm)

    return cm

def cm2rates(cm):
    """Extract true positive/negative and false positive/negative rates from a confusion matrix"""
    dict = Dict()
    n_classes = cm.shape[0]
    for c in range(n_classes):
        tp = cm[c,c] # true positives
        idx = [i for i in range(n_classes) if i != c]
        tn = np.sum(cm[idx,idx]) # true negatives
        fp = np.sum(cm[idx,c]) # false positives
        fn = np.sum(cm[c,idx]) # false negatives
        dict[c] = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    return dict 

def rates2metrics(dict,class_names):
    """
    Computes classification metrics from true positive/negative (tp/tn) and 
    false positive/negative (fp/fn) rates
    Overall metrics are obtained by a weighted average of the class-specific 
    metrics, with weights equal to the class support (number of occurrences)
    Mean metrics are obtained by a non-weighted average of the class-specific 
    metrics
    Args:
        - dict (dictionary): dictionary containing tp, tn, fp, fn values for each class
            (typically obtained with cm2rates())
        - class_names (list of str): class names
    Output:
        - dict_out (dictionary): contains all the computed metrics
    """
    # avoids having to catch zero divisions and 0/0
    div = lambda n, d: n / d if d and n else 0

    dict_out = Dict()
    class_acc = []
    class_prec = []
    class_rec = []
    class_f1 = []
    class_support = []
    # compute support of each class
    for c in dict:
        d = dict[c]
        support = d['tp'] + d['fn']
        class_support.append(support)
    tot_support = sum(class_support)
    # compute metrics for each class
    for c in dict:
        d = dict[c]
        acc = div((d['tp'] + d['tn']) , sum(d.values())) 
        prec = div(d['tp'] , (d['tp'] + d['fp']))
        rec = div(d['tp'] , class_support[c])
        f1 = div(2 * prec * rec , prec + rec)  
        dict_out[class_names[c]] = {'accuracy':acc, 'precision':prec, 
            'recall':rec, 'f1-score':f1, 
            'support (%)':class_support[c]/tot_support*100, 'support':class_support[c]}
        class_acc.append(acc)
        class_prec.append(prec)
        class_rec.append(rec)
        class_f1.append(f1)
    # compute mean metrics
    n_classes = len(class_names)
    dict_out['mean'] = {'accuracy': sum(class_acc)/n_classes,
                        'precision':sum(class_prec)/n_classes, 
                        'recall':sum(class_rec)/n_classes, 
                        'f1-score':sum(class_f1)/n_classes, 
                        'support (%)': 100,
                        'support':sum(class_support)}
    # compute overall metrics                    
    weighted_mean = lambda metric, weight: div(sum([m*w for m,w in zip(metric,weight)]) , sum(weight))
    dict_out['overall'] = {'accuracy': weighted_mean(class_acc, class_support),
                        'precision':weighted_mean(class_prec, class_support), 
                        'recall':weighted_mean(class_rec, class_support), 
                        'f1-score':weighted_mean(class_f1, class_support), 
                        'support (%)': 100,
                        'support':sum(class_support)}
    return dict_out

def get_error_map(pred, target, nodata_val):
    error_map = pred - target
    mask = target == nodata_val
    error_map[mask] = 0
    return error_map
        