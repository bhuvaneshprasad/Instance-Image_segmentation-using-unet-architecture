from instanceSegmentation import logger
from instanceSegmentation.components.model_training import ModelTraining
from instanceSegmentation.config.configuration import ConfigurationManager

STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    """
    A class representing a pipeline for training a model.

    Attributes:
        None
    """
    def __init__(self) -> None:
        """
        Initialize the ModelTrainingPipeline class.
        """
        pass
    
    def main(self):
        """
        Main function to execute the model training pipeline.

        Raises:
            Exception: If there is an error during model training.
        """
        try:
            config = ConfigurationManager()
            model_training_config = config.get_model_training_config()
            model_training = ModelTraining(config=model_training_config)
            model_training.get_base_model()
            model_training.train()
        except Exception as e:
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e