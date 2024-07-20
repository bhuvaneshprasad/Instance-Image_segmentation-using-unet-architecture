from pathlib import Path
from instanceSegmentation.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from instanceSegmentation.utils.common import read_yaml, create_directories
from instanceSegmentation.entity.config_entity import DataIngestionConfig, ModelBuildingConfig

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
    
    def get_model_building_config(self) -> ModelBuildingConfig:
        """
        Get the Model Building configuration.

        Returns:
            ModelBuildingConfig: Model Building configuration settings.
        """
        config = self.config.model_building
        
        model_building_config = ModelBuildingConfig(
            root_dir=Path(config.root_dir),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_augmentation=self.params.AUGMENTATION,
            params_image_height=self.params.IMAGE_HEIGHT,
            params_image_width=self.params.IMAGE_WIDTH,
            params_batch_size=self.params.BATCH_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_num_classes=self.params.NUM_CLASSES
        )
        
        return model_building_config