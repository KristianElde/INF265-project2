from torch.nn import BCEWithLogitsLoss
from torch.nn import MSELoss
from torch.nn import CrossEntropyLoss
import torch


def _compute_detection_loss(y_true, outputs):
    criterion = BCEWithLogitsLoss(reduction="none")
    return criterion(outputs[:, 0], y_true[:, 0])


def _compute_localization_loss(y_true, outputs):
    criterion = MSELoss(reduction="none")
    loss = criterion(outputs[:, 1:5], y_true[:, 1:5])
    loss = torch.sum(loss, dim=1)
    return loss


def _compute_classification_loss(y_true, outputs):
    criterion = CrossEntropyLoss(reduction="none")
    return criterion(outputs[:, 5:], y_true[:, 5])


def localization_loss(outputs, y_true):
    detection_loss = _compute_detection_loss(y_true, outputs)
    localization_loss = _compute_localization_loss(y_true, outputs)
    classification_loss = _compute_classification_loss(y_true, outputs)
    object_present = (y_true[:, 0] == 1).float()
    loss = detection_loss + object_present * (localization_loss + classification_loss)

    return loss


def detection_loss(outputs, y_true):
    batch_size = outputs.shape[0]
    losses = localization_loss(outputs.view(-1, 7), y_true.view(-1, 7))
    loss = torch.sum(losses).item()
    return loss / batch_size
