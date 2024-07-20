import os
from box.exceptions import BoxValueError
import yaml
from instanceSegmentation import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read and load data from a YAML file.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Returns:
        Box: A Box object containing the YAML file data.

    Raises:
        ValueError: If the YAML file is empty or cannot be loaded.
        Exception: For any other unexpected errors during file reading or parsing.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("Yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool=True):
    """
    Create directories specified in the list.

    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool, optional): Whether to log directory creation (default is True).

    Raises:
        Exception: For any unexpected errors during directory creation.
    """
    try:
        for dirs in path_to_directories:
            os.makedirs(dirs, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path_to_directories}")
    except Exception as e:
        raise e

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save JSON data to a file.

    Args:
        path (Path): Path to the JSON file.
        data (dict): Dictionary containing JSON serializable data.

    Raises:
        Exception: For any unexpected errors during file saving.
    """
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Json saved at: {path}")
    except Exception as e:
        raise e

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load JSON data from a file and return as a Box object.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        Box: A Box object containing the JSON data.

    Raises:
        Exception: For any unexpected errors during file loading.
    """
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        logger.info(f"Successfully loaded json from: {path}")
        return ConfigBox(content)
    except Exception as e:
        raise e

@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    """
    Save binary data using joblib.

    Args:
        data (Any): Data to be saved.
        path (Path): Path to save the binary file.

    Raises:
        Exception: For any unexpected errors during file saving.
    """
    try:
        joblib.dump(value=data, filename=path)
        logger.info(f"Binary file saved at: {path}")
    except Exception as e:
        raise e

@ensure_annotations
def load_bin(path: Path):
    """
    Load binary data using joblib.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: Loaded binary data.

    Raises:
        Exception: For any unexpected errors during file loading.
    """
    joblib.load(filename=path)
    logger.info(f"Successfully loaded binary from: {path}")

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get the size of a file in kilobytes.

    Args:
        path (Path): Path to the file.

    Returns:
        str: Size of the file in kilobytes formatted as "{size} kb".

    Raises:
        Exception: For any unexpected errors during file size retrieval.
    """
    try:
        size_in_kb = round(os.path.getsize(path)/1024, 2)
        return f"{size_in_kb} kb"
    except Exception as e:
        raise e

def decodeImage(imgString, fileName) -> None:
    """
    Decode base64 encoded image data and save it to a file.

    Args:
        imgString (str): Base64 encoded image data.
        fileName (str): Name of the file to save the decoded image data.

    Raises:
        Exception: For any unexpected errors during decoding or file writing.
    """
    try:
        imgData = base64.b64decode(imgString)
        with open(fileName, 'wb') as f:
            f.write(imgData)
            f.close
    except Exception as e:
        raise e

def encodeImage(imagePath) -> bytes:
    """
    Encode an image file to base64 bytes.

    Args:
        imagePath (str): Path to the image file.

    Returns:
        bytes: Base64 encoded bytes of the image data.

    Raises:
        Exception: For any unexpected errors during encoding or file reading.
    """
    try:
        with open(imagePath, 'rb') as f:
            imgData = f.read()
            return base64.b64encode(imgData)
    except Exception as e:
        raise e