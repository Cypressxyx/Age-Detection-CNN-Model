from ModeBuilder.model_builder_tools import create_model
class GenderDetectionModel:
    def __init__(self):
        self.model = create_model(255)
        #self.base_model = loaD_base_model(255)
        #self.