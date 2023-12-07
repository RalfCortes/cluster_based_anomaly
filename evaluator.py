from typing import Any

import numpy as np

intervals = list[int]

class TSEvaluator:
    def __init__(self, alpha_recall: int = 0.5, alpha_precision: int = 0) -> None:
        self.alpha_recall = alpha_recall
        self.alpha_precision = alpha_precision

    def find_anomaly_ranges(self, labels: list[int])-> list[intervals]:
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
    
    def overlap_reward(self, anomaly_range: intervals, prediction_range:intervals, cardinality:int)-> tuple[float, int]:
        if anomaly_range[1] < prediction_range[0] or anomaly_range[0] > prediction_range[1]:
            return 0, cardinality
        else:
            cardinality += 1
            start_intersect = max(anomaly_range[0], prediction_range[0])
            end_intersect = min(anomaly_range[1], prediction_range[1])
            overlap =[start_intersect, end_intersect] 
            return self.omega(anomaly_range, overlap), cardinality 

    def omega(self, anomaly_range:intervals, overlap:intervals) -> float:
        size_anomaly = anomaly_range[1] - anomaly_range[0] + 1
        overlap_size = overlap[1] - overlap[0] + 1
        return overlap_size / size_anomaly
        
    def gamma(self, cardinality: int) -> float:
        if cardinality == 0:
            return 1
        return 1 / cardinality
    
    def get_recall(self, real_anomalies: list[int], predicted_anomalies: list[int]) -> float:
        label_ranges = self.find_anomaly_ranges(real_anomalies)
        prediction_ranges = self.find_anomaly_ranges(predicted_anomalies)
        
        recall = 0
        if len(label_ranges) == 0:
            return 0
        for i in range(len(label_ranges)):
            overlap_reward = 0
            cardinality = 0
            range_r = label_ranges[i]
            for j in range(len(prediction_ranges)):
                range_p = prediction_ranges[j]
                overlap_reward_single, cardinality = self.overlap_reward(range_r, range_p, cardinality)
                overlap_reward += overlap_reward_single
            overlap_reward = self.gamma(cardinality) * overlap_reward

            if cardinality > 0:
                existence_reward = 1
            else:
                existence_reward = 0

            recall += self.alpha_recall * existence_reward + (1 - self.alpha_recall) * overlap_reward
        recall /= len(label_ranges)
        return recall

    
    def get_precision(self, real_anomalies: list[int], predicted_anomalies: list[int]) -> float:
        label_ranges = self.find_anomaly_ranges(real_anomalies)
        prediction_ranges = self.find_anomaly_ranges(predicted_anomalies)
        precision = 0
        if len(prediction_ranges) == 0:
            return 0
        for i in range(len(prediction_ranges)):
            range_p = prediction_ranges[i]
            overlap_reward = 0
            cardinality = 0
            for j in range(len(label_ranges)):
                range_r = label_ranges[j]
                overlap_reward_single, cardinality = self.overlap_reward(range_p, range_r, cardinality)
                overlap_reward += overlap_reward_single
            overlap_reward = self.gamma(cardinality) * overlap_reward
            if cardinality > 0:
                existence_reward = 1
            else:
                existence_reward = 0
            precision += self.alpha_precision * existence_reward + (1 - self.alpha_precision) * overlap_reward
        precision /= len(prediction_ranges)
        return precision
        