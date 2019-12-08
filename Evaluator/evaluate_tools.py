"""
 def evaluate_predictions(predictions, actual_classes, labels, class_labels):
 14     predicted_classes = np.argmax(predictions, axis=1)  # speed issues: 3 minutes -> 9 minutes, look into
 15     cm = confusion_matrix(actual_classes, predicted_classes, labels=class_labels)
 16     print_cm(cm, labels=labels)
 17     # print_confusion_matrix(cm, labels=labels) print only if you want a normalize cm in %
 18     f1_scores = evaluate_f_score(actual_classes, predicted_classes, class_labels)
 19     accuracy_scores = accuracy_score(actual_classes, predicted_classes)
 20     return cm, f1_scores, accuracy_scores
 21
 22 # ----------------------------------------------------------------------------
 23 # Evaluate predictions using an F1-Score
 24
 25 def evaluate_f_score(actual_labels, predicted_labels, class_labels):
 26     f1_scores = f1_score(actual_labels, predicted_labels, labels=class_labels, average=None)
 27     print('F1-Scores: ', f1_scores)
 28     tested_groups = np.unique(actual_labels)  # Create a list of groups tested on
 29     f1_scores = remove_untested_classes_in_f1(f1_scores, tested_groups)  # Set untested images to -1 for later metrics
 30     return f1_scores

 rom sklearn.metrics import f1_score, accuracy_score, confusion_matrix
"""
from sklearn.metrics import f1_score

def get_f1_score(predicted_classes, actual_classes, labels):
    f1_scores = f1_score(actual_classes, predicted_classes, labels=labels, average=None)
    print("F1-Scores: ", f1_scores)
    return f1_scores

