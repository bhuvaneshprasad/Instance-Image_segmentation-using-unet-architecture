import os
from pathlib import Path
from instanceSegmentation.utils.common import create_directories
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split
from glob import glob
import cv2
import scipy.io
import numpy as np
from instanceSegmentation.entity.config_entity import ModelTrainingConfig
from instanceSegmentation import logger

class ModelTraining:
    """
    A class to handle model training tasks including loading the model,
    training the model, and saving the trained model.

    Attributes:
        config (ModelTrainingConfig): Configuration for the model training process.
    """
    def __init__(self, config: ModelTrainingConfig) -> None:
        """
        Initialize the ModelTraining class with the given configuration.

        Args:
            config (ModelTrainingConfig): Configuration for the model training process.
        """
        self.config = config
        create_directories([config.root_dir])

    def load_dataset(self, path, split=0.2):
        train_x = sorted(glob(os.path.join(path, "Training", "Images", "*")))
        train_y = sorted(glob(os.path.join(path, "Training", "Categories", "*")))

        split_size = int(split * len(train_x))

        train_x, valid_x = train_test_split(train_x, test_size=split_size, random_state=42)
        train_y, valid_y = train_test_split(train_y, test_size=split_size, random_state=42)

        train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
        train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

        return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    
    def get_colormap(self):
        mat_path = self.config.colormap_path
        colormap = scipy.io.loadmat(mat_path)["colormap"]
        colormap = colormap * 256
        colormap = colormap.astype(np.uint8)
        colormap = [[c[2], c[1], c[0]] for c in colormap]

        classes = self.config.params_classes

        return classes, colormap
    
    def read_image_mask(self, x, y):
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        assert x.shape == y.shape

        x = cv2.resize(x, (self.config.params_image_width, self.config.params_image_height))
        y = cv2.resize(y, (self.config.params_image_width, self.config.params_image_height))

        x = x / 255.0
        x = x.astype(np.float32)
        
        _, colormap = self.get_colormap()

        output = []
        for color in colormap:
            cmap = np.all(np.equal(y, color), axis=-1)
            output.append(cmap)
        output = np.stack(output, axis=-1)
        output = output.astype(np.uint8)

        return x, output
    
    def preprocess(self, x, y):
        def f(x, y):
            x = x.decode()
            y = y.decode()
            image, mask = self.read_image_mask(x, y)
            return image, mask

        image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.uint8])
        image.set_shape([self.config.params_image_height, self.config.params_image_width, 3])
        mask.set_shape([self.config.params_image_height, self.config.params_image_width, self.config.params_num_classes])

        return image, mask

    def tf_dataset(self, x, y, batch=8):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.map(self.preprocess)
        dataset = dataset.batch(batch)
        dataset = dataset.prefetch(2)
        return dataset
        
    def get_base_model(self):
        """
        Load the base model from the specified path in the configuration.
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the trained model to the specified path.

        Args:
            path (Path): Path to save the model.
            model (tf.keras.Model): Trained model to be saved.
        """
        model.save(path)

    def get_dataset(self):
        np.random.seed(42)
        tf.random.set_seed(42)
        
        dataset_path = self.config.dataset_path
        
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = self.load_dataset(dataset_path)
        
        logger.info(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")

        train_dataset = self.tf_dataset(train_x, train_y, batch=self.config.params_batch_size)
        valid_dataset = self.tf_dataset(valid_x, valid_y, batch=self.config.params_batch_size)
        
        return train_dataset, valid_dataset
    
    def train(self):
        """
        Train the model on the training dataset and validate on the validation dataset.
        """
        model_path = self.config.trained_model_path
        csv_path = self.config.csv_path
        
        train_dataset, valid_dataset = self.get_dataset()
        
        callbacks = [
            ModelCheckpoint(model_path, verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
            CSVLogger(csv_path, append=True),
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
        ]
        
        self.model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=self.config.params_epochs,
            callbacks=callbacks
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )