import torch

def calculate_miou(pred_masks, true_masks, num_classes):
    class_iou = []
    epsilon = 1e-8  # Small epsilon value to avoid division by zero

    for class_id in range(num_classes):
        class_pred = (pred_masks == class_id)
        class_true = (true_masks == class_id)
        if torch.sum(class_true).item() > 0:
            intersection = torch.logical_and(class_pred, class_true)
            union = torch.logical_or(class_pred, class_true)

            intersection_sum = torch.sum(intersection).item()
            union_sum = torch.sum(union).item()

            # Check for division by zero
            iou = intersection_sum / (union_sum + epsilon) if union_sum != 0 else 0.0
            class_iou.append(iou)

    mIoU = torch.mean(torch.tensor(class_iou))
    return mIoU.item()



def calculate_class_accuracy(pred_masks, true_masks, num_classes):
    class_accuracy = []
    for class_id in range(num_classes):
        
        class_pred = (pred_masks == class_id)
        class_true = (true_masks == class_id)
        if torch.sum(class_true).item() > 0:
            correct_pixels = torch.sum(class_pred == class_true).item()
            total_pixels = true_masks.size(0)*true_masks.size(1)*true_masks.size(2)
            acc = correct_pixels / (total_pixels) 

            class_accuracy.append(acc)
    mean_acc = torch.mean(torch.tensor(class_accuracy))
    return mean_acc.item()


def calculate_precision(pred_masks, true_masks, num_classes):
    class_precision = []
    epsilon = 1e-8  # Small epsilon value to avoid division by zero

    for class_id in range(num_classes):
        class_pred = (pred_masks == class_id)
        class_true = (true_masks == class_id)

        if torch.sum(class_true).item() > 0:
            true_positive = torch.sum(torch.logical_and(class_pred, class_true)).item()
            false_positive = torch.sum(class_pred).item() - true_positive

            # Check for division by zero
            precision = true_positive / (true_positive + false_positive + epsilon) if (true_positive + false_positive) != 0 else 0.0
            class_precision.append(precision)

    mean_precision = torch.mean(torch.tensor(class_precision))
    return mean_precision.item()

def calculate_recall(pred_masks, true_masks, num_classes):
    class_recall = []
    epsilon = 1e-8  # Small epsilon value to avoid division by zero

    for class_id in range(num_classes):
        class_pred = (pred_masks == class_id)
        class_true = (true_masks == class_id)
        if torch.sum(class_true).item() > 0:
            true_positive = torch.sum(torch.logical_and(class_pred, class_true)).item()
            false_negative = torch.sum(class_true).item() - true_positive

            # Check for division by zero
            recall = true_positive / (true_positive + false_negative + epsilon) if (true_positive + false_negative) != 0 else 0.0
            class_recall.append(recall)

    mean_recall = torch.mean(torch.tensor(class_recall))
    return mean_recall.item()
