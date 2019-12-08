"""
A Class definition for the Gender model
"""

from ModelBuilder.model_builder_tools import create_model

class GenderDetectionModel:
    def __init__(self):
        self.model = create_model(255)
        self.batch_size = 32
        self.num_epochs = 8
        self.params = dict( "batch_size": self.batch_size,
                            "epochs": self.num_epochs)

    """ Train the CNN model"""
    def train_model(self, train_data, train_labels):
        self.model.fit(train_data, train_labels, **self.params)

    """ Test the CNN model"""
    def test_model(self, test_data, test_labels):
        print("------ Model Loss and Model Accuracy --------")
        print(str(self.model.evaluate(test_data, test_labels)))
        return self.predict_classes(test_data)

    """ Predict the classes of the diven data"""
    def predict_classes(self, data):
        return self.model.predict(data)
