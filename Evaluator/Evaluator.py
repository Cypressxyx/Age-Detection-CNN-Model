"""
Will evaluate our scores
 def evaluate_preds(self, preds, non_encoded_test_labels):
 55         self.cm, self.f1_scores, self.accuracy_score = evaluate_predictions(preds, non_encoded_test_labels,
 56                                                                             self.classes, self.class_labels)
 57         self.all_cm.append(self.cm)
 58         self.all_predictions.extend(preds)
 59         self.prediction = preds
 60         self.accuracy_scores.append(self.accuracy_score)
 61         self.split_metrics[self.split_key] = [self.accuracy_score, self.f1_scores, self.cm]
 62         self.split_key += 1
"""

from Evaluator.evaluate_tools import get_f1_score
class Evaluator:
    def __init__(self, labels):
        self.score = 0
        self.labels = ["male", "female"]

    def evaluate_predictions(self, predictions train_labels, ):
        predicted_classes = np.argmax(predictions, axis=1) 
        self.f1_score = get_f1_score(predicted_classes, train_labels, self.labels)

        
