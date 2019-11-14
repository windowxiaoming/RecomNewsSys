#-*-coding:utf-8-*-
import numpy as np
import scipy.stats as st
FLOAT_MAX = np.finfo(np.float32).max

def mrr_score(model, test, train=None):
    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    mrrs = []
    for user_id, row in enumerate(test):
        if not len(row.indices):
            continue
        predictions = -model.predict(user_id)
        if train is not None:
            predictions[train[user_id].indices] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[row.indices]).mean()
        mrrs.append(mrr)

    return np.array(mrrs)

def sequence_mrr_score(model, test, exclude_preceding=False):
    sequences = test.sequences[:, :-1]
    targets = test.sequences[:, -1:]
    mrrs = []
    for i in range(len(sequences)):
        predictions = -model.predict(sequences[i])
        if exclude_preceding:
            predictions[sequences[i]] = FLOAT_MAX
        mrr = (1.0 / st.rankdata(predictions)[targets[i]]).mean()
        mrrs.append(mrr)

    return np.array(mrrs)


def sequence_precision_recall_score(model, test, k=10, exclude_preceding=False):
    sequences = test.sequences[:, :-k]
    targets = test.sequences[:, -k:]
    precision_recalls = []
    for i in range(len(sequences)):
        predictions = -model.predict(sequences[i])
        if exclude_preceding:
            predictions[sequences[i]] = FLOAT_MAX

        predictions = predictions.argsort()[:k]
        precision_recall = _get_precision_recall(predictions, targets[i], k)
        precision_recalls.append(precision_recall)

    precision = np.array(precision_recalls)[:, 0]
    recall = np.array(precision_recalls)[:, 1]
    return precision, recall


def _get_precision_recall(predictions, targets, k):

    predictions = predictions[:k]
    num_hit = len(set(predictions).intersection(set(targets)))
    return float(num_hit) / len(predictions), float(num_hit) / len(targets)

def precision_recall_score(model, test, train=None, k=10):
    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if np.isscalar(k):
        k = np.array([k])

    precision = []
    recall = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            rated = train[user_id].indices
            predictions[rated] = FLOAT_MAX

        predictions = predictions.argsort()

        targets = row.indices

        user_precision, user_recall = zip(*[
            _get_precision_recall(predictions, targets, x)
            for x in k
        ])

        precision.append(user_precision)
        recall.append(user_recall)

    precision = np.array(precision).squeeze()
    recall = np.array(recall).squeeze()
    return precision, recall


def rmse_score(model, test):
    predictions = model.predict(test.user_ids, test.item_ids)
    return np.sqrt(((test.ratings - predictions) ** 2).mean())
