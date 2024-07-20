from instanceSegmentation.components.data_ingestion import DataIngestion
from instanceSegmentation.config.configuration import ConfigurationManager
from instanceSegmentation import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionPipeline:
    """
    A class representing a pipeline for data ingestion.

    Attributes:
        None
    """
    def __init__(self) -> None:
        """
        Initialize the DataIngestionPipeline class.
        """
        pass
    
    def main(self):
        """
        Main function to execute the data ingestion pipeline.

        Raises:
            Exception: If there is an error during data ingestion.
        """
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
        except Exception as e:
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e