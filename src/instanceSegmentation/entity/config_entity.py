from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    data_dir: Path

@dataclass(frozen=True)
class ModelBuildingConfig:
    root_dir: Path
    updated_base_model_path: Path
    params_augmentation: bool
    params_image_height: int
    params_image_width: int
    params_batch_size: int
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: list
    params_num_classes: int

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    updated_base_model_path: Path
    trained_model_path: Path
    colormap_path: Path
    params_classes: list
    params_image_height: int
    params_image_width: int
    params_num_classes: int
    params_batch_size: int
    params_epochs: int
    dataset_path: Path
    csv_path: Path

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    score_csv_path: Path
    trained_model_path: Path
    training_data_path: Path
    params_image_height: int
    params_image_width: int
    colormap_path: Path
    dataset_path: Path
    params_classes: list
    