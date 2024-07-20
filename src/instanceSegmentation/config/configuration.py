from instanceSegmentation.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from instanceSegmentation.utils.common import read_yaml, create_directories
from instanceSegmentation.entity.config_entity import DataIngestionConfig

class ConfigurationManager:
    """
    A class to manage the configuration settings for the Instance Segmentation.

    Attributes:
        config_filepath (str): Path to the configuration file.
        params_filepath (str): Path to the parameters file.
    """
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH) -> None:
        """
        Initialize the ConfigurationManager with paths to the configuration and parameters files.

        Args:
            config_filepath (str): Path to the configuration file.
            params_filepath (str): Path to the parameters file.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get the Data Ingestion configuration.

        Returns:
            DataIngestionConfig: Data ingestion configuration settings.
        """
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            data_dir=config.data_dir
        )
        
        return data_ingestion_config