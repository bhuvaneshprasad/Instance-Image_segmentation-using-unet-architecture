from instanceSegmentation import logger
from instanceSegmentation.components.model_evaluation import ModelEvaluation
from instanceSegmentation.config.configuration import ConfigurationManager
from instanceSegmentation.constants import CONFIG_FILE_PATH
from instanceSegmentation.utils.common import read_yaml

STAGE_NAME = "Model Evaluation"

class ModelEvaluationPipeline:
    """
    A class representing a pipeline for evaluating a model.

    Attributes:
        None
    """
    def __init__(self) -> None:
        """
        Initialize the ModelEvaluationPipeline class.
        """
        pass
    
    def main(self):
        """
        Main function to execute the model evaluation pipeline.

        Raises:
            Exception: If there is an error during model evaluation or logging.
        """
        try:
            config = read_yaml(CONFIG_FILE_PATH)
            config = ConfigurationManager()
            eval_config = config.get_model_evaluation_config()
            eval = ModelEvaluation(eval_config)
            eval.evaluation()
        except Exception as e:
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e