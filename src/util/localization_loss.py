from torch.nn import BCEWithLogitsLoss
from torch.nn import MSELoss
from torch.nn import CrossEntropyLoss
import torch


def _compute_detection_loss(Y_true, outputs):
    criterion = BCEWithLogitsLoss(reduction="none")
    return criterion(outputs[:, 0], Y_true[:, 0])


def _compute_localization_loss(Y_true, outputs):
    criterion = MSELoss(reduction="none")
    loss = criterion(outputs[:, 1:5], Y_true[:, 1:5])
    loss = torch.mean(loss, dim=(1))
    return loss


def _compute_classification_loss(Y_true, outputs):
    criterion = CrossEntropyLoss(reduction="none")
    target = torch.argmax(Y_true[:, 5:], dim=1)
    return criterion(outputs[:, 5:], target)


def localization_loss(outputs, Y_true):
    detection_loss = _compute_detection_loss(Y_true, outputs)
    localization_loss = _compute_localization_loss(Y_true, outputs)
    classification_loss = _compute_classification_loss(Y_true, outputs)
    object_present = (Y_true[:, 0] == 1).float()
    loss = detection_loss + object_present * (localization_loss + classification_loss)

    return torch.mean(loss)
