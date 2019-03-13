def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, accuracy, f1 - classification metrics
    '''
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for p, t in zip(prediction, ground_truth):
        if p == t and p == True:
            TP += 1
        elif p == t and p == False:
            TN += 1
        elif p == True:
            FP += 1
        else:
            FN += 1
    precision = TP / (TP + FP) if TP + FP != 0 else 1
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / len(prediction) 
    f1 = 2 * (precision * recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1,accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    
    # TODO: Implement computing accuracy
    return (prediction==ground_truth).sum() / prediction.shape[0]
