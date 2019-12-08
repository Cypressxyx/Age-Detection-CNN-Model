"""
@description: A CNN Model that can detect Gender
@Dataset:
    Can be found at https://github.com/JingchunCheng/All-Age-Faces-Dataset
""" 
from DataLoader.DataLoader import Dataset
from Evaluator.Evaluator import Evaluator
from ModelBuilder.ModelBuilder import AgeDetectionModel


"""  Collect the data to train and test """
dataset      = Dataset('./faces')
train, test  = dataset.get_train_test_data()
test_data, test_labels = train
train_data, train_labels = train

""" Train the model and test the model """
gender_model = GenderDetectionModel()
gender_model.train(train_data, train_labels)

""" Evaluate the model """
evaluator   = Evaluator()
predictions = gender_model.test_model(test_data, test_labels)
evaluator.evaluate_predictions(predictions, test_labels)