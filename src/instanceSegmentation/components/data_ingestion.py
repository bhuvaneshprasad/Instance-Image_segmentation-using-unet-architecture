import os
import opendatasets as od
from instanceSegmentation.entity.config_entity import DataIngestionConfig
from instanceSegmentation import logger

class DataIngestion:
    """
    A class to handle data ingestion tasks such as downloading files.

    Attributes:
        config (DataIngestionConfig): Configuration for data ingestion.
    """
    def __init__(self, config: DataIngestionConfig) -> None:
        """
        Initialize the DataIngestion class with the given configuration.

        Args:
            config (DataIngestionConfig): Configuration for data ingestion.
        """
        self.config = config
    
    def download_file(self) -> None:
        """
        Download a file from the source URL to the specified data directory.

        Raises:
            Exception: If there is any error during the download process.
        """
        try:
            dataset_url = self.config.source_url
            download_dir = self.config.data_dir
            os.makedirs(download_dir, exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {download_dir}")
            
            od.download(dataset_url, data_dir=download_dir)
            logger.info(f"Downloaded data from {dataset_url} into file {download_dir}")
        except Exception as e:
            raise e