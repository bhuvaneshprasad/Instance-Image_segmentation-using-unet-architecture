from instanceSegmentation.components.model_building import ModelBuilding
from instanceSegmentation.config.configuration import ConfigurationManager
from instanceSegmentation import logger

STAGE_NAME = "Prepare Base Model Stage"

class ModelBuildingPipeline:
    """
    A class representing a pipeline for preparing the model.

    Attributes:
        None
    """
    def __init__(self) -> None:
        """
        Initialize the PrepareBaseModelPipeline class.
        """
        pass
    
    def main(self):
        """
        Main function to execute the model building pipeline.

        Raises:
            Exception: If there is an error during model preparation.
        """
        try:
            config = ConfigurationManager()
            prepare_base_model_config = config.get_model_building_config()
            prepare_base_model = ModelBuilding(config=prepare_base_model_config)
            prepare_base_model.update_base_model()
        except Exception as e:
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelBuildingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e