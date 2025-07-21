import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

def eval_map(results, gt_labels, ind2cat):
    results = np.asarray(results)
    gt_labels = np.asarray(gt_labels)

    results = results.transpose()
    gt_labels = gt_labels.transpose()

    assert results.shape[0] == gt_labels.shape[0]
    ap = np.zeros(26, dtype=np.float32)
    for i in range(26):
        ap[i] = average_precision_score(gt_labels[i, :], results[i, :])
        print('Category %16s %.5f' % (ind2cat[i], ap[i]))
    print('Mean AP %.5f' % (ap.mean()))
    return ap


def eval_acc(results, gt_labels):
    correct = 0
    total = len(results)
    for i in range(total):
        if gt_labels[i] == results[i]:
            correct += 1
    acc = correct / total
    return acc


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
    ap = np.zeros(26, dtype=np.float32)
    for i in range(26):
        ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
        print('Category %16s %.5f' % (ind2cat[i], ap[i]))
    print('Mean AP %.5f' % (ap.mean()))
    return ap



def get_thresholds(cat_preds, cat_labels):
    thresholds = np.zeros(26, dtype=np.float32)
    for i in range(26):
        p, r, t = precision_recall_curve(cat_labels[i, :], cat_preds[i, :])
        for k in range(len(p)):
            if p[k] == r[k]:
                thresholds[i] = t[k]
                break
    return thresholds


def get_class_by_dataset(dataset):
    if dataset == 'EMOTIC':
        classes = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval',
                   'Disconnection', \
                   'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue',
                   'Fear', 'Happiness', \
                   'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy',
                   'Yearning']

    if dataset == 'CAER-S':
        classes = ['Disgust', 'Anger', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    if dataset == 'HECO':
        classes = ['Surprise', 'Excitement', 'Happiness', 'Peace', 'Disgust', 'Anger', 'Fear', 'Sadness']

    if dataset == 'Emotion6':
        classes = ['surprise', 'sadness', 'joy', 'fear', 'disgust', 'anger']

    if dataset == 'FI':
        classes = ['sadness', 'fear', 'excitement', 'disgust', 'contentment', 'awe', 'anger', 'amusement']

    if dataset == 'UBE':
        classes = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    if dataset == 'emo8':
        classes = ['sadness', 'fear', 'excitement', 'disgust', 'contentment', 'awe', 'anger', 'amusement']

    return classes


def get_class_num_by_dataset(dataset):
    if dataset == 'EMOTIC':
        num_classes = 26
    if dataset == 'HECO':
        num_classes = 8
    if dataset == 'FI':
        num_classes = 8
    if dataset == 'emo8':
        num_classes = 8
    if dataset == 'CAER-S':
        num_classes = 7
    if dataset == 'Emotion6':
        num_classes = 6
    if dataset == 'UBE':
        num_classes = 6

    return num_classes