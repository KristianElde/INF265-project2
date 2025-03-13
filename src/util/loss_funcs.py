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
    loss = torch.mean(loss, dim=(1))
    return loss


def _compute_classification_loss(y_true, outputs):
    criterion = CrossEntropyLoss(reduction="none")
    target = torch.argmax(y_true[:, 5:], dim=1)
    return criterion(outputs[:, 5:], target)


def localization_loss(outputs, y_true):
    detection_loss = _compute_detection_loss(y_true, outputs)
    localization_loss = _compute_localization_loss(y_true, outputs)
    classification_loss = _compute_classification_loss(y_true, outputs)
    object_present = (y_true[:, 0] == 1).float()
    loss = detection_loss + object_present * (localization_loss + classification_loss)

    return loss


def _localization_loss_single(outputs, y_true):
    losses = localization_loss(torch.flatten(outputs), torch.flatten(y_true))
    return torch.sum(losses)
    # Hout * Wout * 7


def detection_loss(outputs, y_true):
    losses = torch.stack(
        [_localization_loss_single(o, y) for o, y in zip(outputs, y_true)]
    )
    return torch.mean(losses)


def detection_loss(outputs, y_true):
    batch_size = outputs.shape[0]
    losses = localization_loss(
        outputs.view(batch_size, -1), y_true.view(batch_size, -1)
    )
    losses = torch.sum(losses, dim=1)  # Sum per sample
    return torch.mean(losses)


def chat_research_detection_loss(predictions, targets):
    # Compute component losses for all samples and all grid cells at once
    det_loss = _compute_detection_loss(
        predictions, targets
    )  # objectness/confidence loss per cell
    loc_loss = _compute_localization_loss(
        predictions, targets
    )  # localization (bbox) loss per cell
    cls_loss = _compute_classification_loss(
        predictions, targets
    )  # classification loss per cell

    # Combine the losses for each grid cell (and anchor, if any)
    total_loss = det_loss + loc_loss + cls_loss

    # Sum over all grid cells (and anchors) for each sample in the batch.
    # We flatten all non-batch dimensions into one and then sum along that dimension.
    loss_per_sample = total_loss.view(total_loss.size(0), -1).sum(dim=1)

    # Compute the mean loss over the batch
    mean_loss = loss_per_sample.mean()
    return mean_loss
