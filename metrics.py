import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def compute_metrics(predicted_labels, gold_labels, negative_class_weight=1):
    predicted_labels = [label.item() for label in predicted_labels]
    gold_labels = [label.item() for label in gold_labels]

    assert len(predicted_labels) == len(gold_labels)

    precision = precision_score(gold_labels, predicted_labels, zero_division=0)
    recall = recall_score(gold_labels, predicted_labels, zero_division=0)
    f1 = f1_score(gold_labels, predicted_labels, zero_division=0)
    accuracy = accuracy_score(gold_labels, predicted_labels) 

    # make the sample_weights array the same size as the number of samples in the dataset, with 1 for positive and weight for negative
    sample_weights = np.ones(len(gold_labels))
    sample_weights[np.array(gold_labels) == 0] = negative_class_weight
    weighted_f1 = f1_score(gold_labels, predicted_labels, sample_weight=sample_weights, zero_division=0)

    return precision, recall, f1, accuracy, weighted_f1