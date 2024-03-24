

def f1_score_masks(y_true, y_pred, pos_label=1):
    """
    Compute the F1 score.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: F1 score.
    """
    true_positives = sum((y_true == pos_label) & (y_pred == pos_label))
    false_positives = sum((y_true != pos_label) & (y_pred == pos_label))
    false_negatives = sum((y_true == pos_label) & (y_pred != pos_label))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision == 0 and recall == 0:
        return 0

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1
