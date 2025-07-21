import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

def vadcat():
    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval',
           'Disconnection', \
           'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue',
           'Fear', 'Happiness', \
           'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion

    vad = ['Valence', 'Arousal', 'Dominance']
    ind2vad = {}
    for idx, continuous in enumerate(vad):
        ind2vad[idx] = continuous

    return ind2cat, ind2vad

def test_scikit_ap(cat_preds, cat_labels, ind2cat):
    ''' Calculate average precision per emotion category using sklearn library.
    :param cat_preds: Categorical emotion predictions.
    :param cat_labels: Categorical emotion labels.
    :param ind2cat: Dictionary converting integer index to categorical emotion.
    :return: Numpy array containing average precision per emotion category.
    '''
    ap = np.zeros(26, dtype=np.float32)
    for i in range(26):
        ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
        print('Category %16s %.5f' % (ind2cat[i], ap[i]))
    print('Mean AP %.5f' % (ap.mean()))
    return ap



def get_thresholds(cat_preds, cat_labels):
    ''' Calculate thresholds where precision is equal to recall. These thresholds are then later for inference.
    :param cat_preds: Categorical emotion predictions.
    :param cat_labels: Categorical emotion labels.
    :return: Numpy array containing thresholds per emotion category where precision is equal to recall.
    '''
    thresholds = np.zeros(26, dtype=np.float32)
    for i in range(26):
        p, r, t = precision_recall_curve(cat_labels[i, :], cat_preds[i, :])
        for k in range(len(p)):
            if p[k] == r[k]:
                thresholds[i] = t[k]
                break
    return thresholds

class DiscreteLoss(nn.Module):
    ''' Class to measure loss between categorical emotion predictions and labels.'''

    def __init__(self, weight_type='mean', device=torch.device('cpu')):
        super(DiscreteLoss, self).__init__()
        self.weight_type = weight_type
        self.device = device
        if self.weight_type == 'mean':
            self.weights = torch.ones((1, 26)) / 26.0
            self.weights = self.weights.to(self.device)
        elif self.weight_type == 'static':
            self.weights = torch.FloatTensor([0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                                              0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                                              0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520,
                                              0.1537]).unsqueeze(0)
            self.weights = self.weights.to(self.device)

    def forward(self, pred, target):
        if self.weight_type == 'dynamic':
            self.weights = self.prepare_dynamic_weights(target)
            self.weights = self.weights.to(self.device)
        loss = (((pred - target) ** 2) * self.weights)
        return loss.sum()

    def prepare_dynamic_weights(self, target):
        target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
        # print('target_stats', target_stats)
        weights = torch.zeros((1, 26))
        weights[target_stats != 0] = 1.0 / torch.log(target_stats[target_stats != 0].data + 1.2)
        weights[target_stats == 0] = 0.0001
        # print('weights', weights)
        return weights


