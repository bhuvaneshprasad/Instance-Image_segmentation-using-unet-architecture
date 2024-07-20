from dotenv import load_dotenv
from instanceSegmentation import logger
from instanceSegmentation.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from instanceSegmentation.pipeline.stage_02_model_building import ModelBuildingPipeline
from src.instanceSegmentation.pipeline.stage_03_model_training import ModelTrainingPipeline

load_dotenv()

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Building Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = ModelBuildingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = " Model Training Pipeline"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e