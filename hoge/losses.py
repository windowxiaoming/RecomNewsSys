#-*-coding:utf-8-*-
import torch
import torch.nn.functional as F
from hoge.torch_utils import assert_no_grad

def pointwise_loss(positive_predictions, negative_predictions, mask=None):
    positives_loss = (1.0 - torch.sigmoid(positive_predictions))
    negatives_loss = torch.sigmoid(negative_predictions)

    loss = (positives_loss + negatives_loss)

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()


def bpr_loss(positive_predictions, negative_predictions, mask=None):
    loss = (1.0 - torch.sigmoid(positive_predictions -
                            negative_predictions))

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()


def hinge_loss(positive_predictions, negative_predictions, mask=None):
    loss = torch.clamp(negative_predictions -
                       positive_predictions +
                       1.0, 0.0)

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()


def adaptive_hinge_loss(positive_predictions, negative_predictions, mask=None):
    highest_negative_predictions, _ = torch.max(negative_predictions, 0)
    return hinge_loss(positive_predictions, highest_negative_predictions.squeeze(), mask=mask)


def regression_loss(observed_ratings, predicted_ratings):
    assert_no_grad(observed_ratings)
    return ((observed_ratings - predicted_ratings) ** 2).mean()


def poisson_loss(observed_ratings, predicted_ratings):
    assert_no_grad(observed_ratings)

    return (predicted_ratings - observed_ratings * torch.log(predicted_ratings)).mean()


def logistic_loss(observed_ratings, predicted_ratings):
    assert_no_grad(observed_ratings)
    observed_ratings = torch.clamp(observed_ratings, 0, 1)
    return F.binary_cross_entropy_with_logits(predicted_ratings,
                                              observed_ratings,
                                              size_average=True)
