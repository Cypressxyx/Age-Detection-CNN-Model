from ModelBuilder.model_builder_tools import create_model

class GenderDetectionModel:
    def __init__(self):
        self.model = create_model(255)