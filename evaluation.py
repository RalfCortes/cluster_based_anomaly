import numpy as np
    
def find_anomaly_ranges(labels):
    ranges = []
    start = None
    for i, label in enumerate(labels):
        if label == 0 and start is not None:
            ranges.append([start, i - 1])
            start = None
        elif label == 1 and start is None:
            start = i
    if start is not None:
        ranges.append([start, len(labels) - 1])
    return ranges

def compute_intersection_union_lists(labels, prediction):
    intersection_labels = np.zeros(len(labels))
    union_labels = np.zeros(len(labels))
    intersection_prediction = np.zeros(len(prediction))
    union_prediction = np.zeros(len(prediction))
    for i, range_labels in enumerate(labels):
        for j, range_prediction in enumerate(prediction):
            start_union = min(range_labels[0], range_prediction[0])
            end_union = max(range_labels[1], range_prediction[1])
            intersection_start = max(range_labels[0], range_prediction[0])
            intersection_end = min(range_labels[1], range_prediction[1])
            overlap = intersection_end - intersection_start + 1
            union_range = end_union - start_union
            if overlap > 0:
                intersection_labels[i] += overlap
                union_labels[i] += union_range
                intersection_prediction[j] += overlap
                union_prediction[j] += union_range
    return intersection_labels, union_labels, intersection_prediction, union_prediction

def existence_score(intersection_labels):
    return np.mean(intersection_labels > 0)

def overlap_score(intersection, union):
    union[union == 0] = 1
    return np.mean(intersection / union)

def compute_metrics(labels, prediction):
    ranges_labels = find_anomaly_ranges(labels)
    ranges_prediction = find_anomaly_ranges(prediction)
    intersection_labels, union_labels, intersection_prediction, union_prediction = compute_intersection_union_lists(ranges_labels, ranges_prediction)
    existence = existence_score(intersection_labels)
    overlap = overlap_score(intersection_labels, union_labels)
    precision = overlap_score(intersection_prediction, union_prediction)
    return {"existence": existence, "overlap": overlap, "precision": precision}