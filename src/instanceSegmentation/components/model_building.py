from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from instanceSegmentation.entity.config_entity import ModelBuildingConfig
from instanceSegmentation.utils.common import create_directories

class ModelBuilding:
    """
    A class to prepare and save the model for training.

    Attributes:
        config (ModelBuildingConfig): Configuration for preparing the model.
    """
    def __init__(self, config: ModelBuildingConfig) -> None:
        """
        Initialize the PrepareBaseModel class with the given configuration.

        Args:
            config (PrepareBaseModelConfig): Configuration for preparing the base model.
        """
        self.config = config
        create_directories([self.config.root_dir])
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the model to the specified path.

        Args:
            path (Path): Path to save the model.
            model (tf.keras.Model): Model to be saved.
        """
        model.save(path)
    
    def conv_block(self, input, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x
    
    def decoder_block(self, input, skip_features, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
        x = Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x
    
    @staticmethod
    def _build_unet(self, input_shape, num_classes):
        inputs = Input(input_shape)
        
        # Load MobileNetV2 as the encoder
        mobile_net = MobileNetV2(include_top=False, weights=self.config.params_weights, input_tensor=inputs)
        
        # Define skip connections from MobileNetV2
        skip_connections = [
            mobile_net.get_layer("block_1_expand_relu").output,  # 64 filters
            mobile_net.get_layer("block_3_expand_relu").output,  # 96 filters
            mobile_net.get_layer("block_6_expand_relu").output,  # 144 filters
            mobile_net.get_layer("block_13_expand_relu").output  # 384 filters
        ]

        # Bottom of the U-Net
        bottom = mobile_net.get_layer("block_16_project").output  # 1280 filters

        # Decoder
        d1 = self.decoder_block(bottom, skip_connections[3], 512)
        d2 = self.decoder_block(d1, skip_connections[2], 256)
        d3 = self.decoder_block(d2, skip_connections[1], 128)
        d4 = self.decoder_block(d3, skip_connections[0], 64)

        # Adjust the output layer to ensure the final output shape matches the input shape
        x = Conv2DTranspose(32, (2, 2), strides=2, padding="same")(d4)
        x = self.conv_block(x, 32)

        outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(x)

        self.model = Model(inputs, outputs, name="U-Net_MobileNetV2")
        
        return self.model

    def update_base_model(self):
        """
        Update the base model by adding custom layers and compiling it, then save the updated model.
        """
        self.full_model = self._build_unet(self,
            input_shape=(self.config.params_image_height, self.config.params_image_width, 3), num_classes=self.config.params_num_classes
        )
        
        self.full_model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(self.config.params_learning_rate),
            metrics=['accuracy']
        )
        
        self.full_model.summary()
        
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)