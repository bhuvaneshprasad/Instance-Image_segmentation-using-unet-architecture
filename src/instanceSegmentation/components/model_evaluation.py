import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from instanceSegmentation.entity.config_entity import ModelEvaluationConfig
from instanceSegmentation.utils.common import create_directories
from sklearn.metrics import f1_score, jaccard_score
import cv2
from sklearn.model_selection import train_test_split
from glob import glob
import scipy.io


class ModelEvaluation:
    """
    A class to handle model evaluation tasks including loading the model,
    evaluating on validation data, and saving scores.

    Attributes:
        config (EvaluationConfig): Configuration for the evaluation process.
    """
    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize the Evaluation class with the given configuration.

        Args:
            config (EvaluationConfig): Configuration for the evaluation process.
        """
        self.config = config
        create_directories([self.config.root_dir])

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Load a TensorFlow model from the specified path.
        
        Args:
            path (Path): Path to the model.
        
        Returns:
            tf.keras.Model: Loaded TensorFlow model.
        """
        return tf.keras.models.load_model(path)
    
    @staticmethod
    def load_dataset(path, split=0.2):
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
    
    def grayscale_to_rgb(self, mask, colormap):
        h, w, _ = mask.shape
        mask = mask.astype(np.int32)
        output = []
        for _, pixel in enumerate(mask.flatten()):
            output.append(colormap[pixel])
        output = np.reshape(output, (h, w, 3))
        return output

    def save_results(self, image, mask, pred, save_image_path):
        h, w, _ = image.shape
        line = np.ones((h, 10, 3)) * 255
        
        classes, colormap = self.get_colormap()

        pred = np.expand_dims(pred, axis=-1)
        pred = self.grayscale_to_rgb(pred, colormap)

        cat_images = np.concatenate([image, line, mask, line, pred], axis=1)
        cv2.imwrite(save_image_path, cat_images)
    
    def evaluation(self):
        """
        Evaluate the model on the validation dataset and save the evaluation score.
        """
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = self.load_dataset(self.config.dataset_path)
        
        self.model = self.load_model(self.config.trained_model_path)
        
        SCORE = []
        for x, y in zip(test_x, test_y):
            name = x.split("/")[-1].split(".")[0]
            
            image = cv2.imread(x, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (self.config.params_image_width, self.config.params_image_height))
            image_x = image
            image = image/255.0
            image = np.expand_dims(image, axis=0)
            
            mask = cv2.imread(y, cv2.IMREAD_COLOR)
            mask = cv2.resize(mask, (self.config.params_image_width, self.config.params_image_height))
            mask_x = mask
            onehot_mask = []
            
            classes, colormap = self.get_colormap()
            
            for color in colormap:
                cmap = np.all(np.equal(mask, color), axis=-1)
                onehot_mask.append(cmap)
            onehot_mask = np.stack(onehot_mask, axis=-1)
            onehot_mask = np.argmax(onehot_mask, axis=-1)
            onehot_mask = onehot_mask.astype(np.int32)
            
            pred = self.model.predict(image, verbose=0)[0]
            pred = np.argmax(pred, axis=-1)
            pred = pred.astype(np.float32)
            
            save_image_path = os.path.join(self.config.root_dir, 'results', f"{name}.png")
            self.save_results(image_x, mask_x, pred, save_image_path)
            
            onehot_mask = onehot_mask.flatten()
            pred = pred.flatten()
            
            labels = [i for i in range(len(classes))]
            
            f1_value = f1_score(onehot_mask, pred, labels=labels, average=None, zero_division=0)
            jac_value = jaccard_score(onehot_mask, pred, labels=labels, average=None, zero_division=0)
            
            SCORE.append([f1_value, jac_value])
        self.save_score(self, SCORE, classes)
    
    def save_score(self, SCORE, classes):
        """
        Save the evaluation score to a CSV file.
        """
        score = np.array(SCORE)
        score = np.mean(score, axis=0)
        
        f = open(self.config.score_csv_path, "w")
        f.write("Class,F1,Jaccard\n")
        
        l = ["Class", "F1", "Jaccard"]
        print(f"{l[0]:15s} {l[1]:10s} {l[2]:10s}")
        print("-"*35)
        
        for i in range(score.shape[1]):
            class_name = classes[i]
            f1 = score[0, i]
            jac = score[1, i]
            dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
            print(dstr)
            f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")
        
        print("-"*35)
        class_mean = np.mean(score, axis=-1)
        class_name = "Mean"
        f1 = class_mean[0]
        jac = class_mean[1]
        dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
        print(dstr)
        f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")
        f.close()