import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa
import sys
#from tensorflow.keras import keras 

import warnings

warnings.filterwarnings('ignore')
num_classes = 100
#input_shape = (32, 32, 3)
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 1
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024] 

#Check the dataset
if sys.argv[2] == 'cifar10':
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
elif sys.argv[2] == 'cifar100':
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
elif sys.argv[2] == 'fashion_mnist':
  (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

size = sys.argv[1]
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        return {"patch_size": self.patch_size}

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

#Multilayer Perceptron layers
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim=projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    def get_config(self):
        return {"num_patches": self.num_patches, "projection_dim": self.projection_dim}
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def CreateTransformer(size, outputNeurons, shape, x_train):
    if size == "tiny":
        transformer_layers = 4
    elif size == "small":
        transformer_layers = 6
    elif size == "base":
        transformer_layers = 8
    
    #Data augmentation
    data_augmentation = keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.Normalization(),
        tf.keras.layers.experimental.preprocessing.Resizing(image_size, image_size),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.02),
        tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
    )
    # Compute the mean and the variance of the training data for normalization.
    data_augmentation.layers[0].adapt(x_train)
    
    #set input layer/shape
    inputs = layers.Input(shape=shape)

    # Augment data
    augmented = data_augmentation(inputs)

    # Create patches
    patches = Patches(patch_size)(augmented)

    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Final normalization/output
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    #add mlp to transformer
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    #pass features from mlp to final dense layer/classification
    logits = layers.Dense(outputNeurons)(features)

    #create model
    model2 = keras.Model(inputs=inputs, outputs=logits)

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model2.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    model2.summary()
    return model2

#Load the models
cnn_model = tf.keras.models.load_model(size+"_CNN"+".h5")
trans_model = tf.keras.models.load_model(size+"_vision_transformer"+".h5", custom_objects={"PatchEncoder": PatchEncoder,"Patches": Patches})
#Models prediction
cnn_pred = np.argmax(cnn_model.predict(x_test), axis=1)
trans_pred = np.argmax(trans_model.predict(x_test), axis=1)
#Print the performance
print("CNN model performance is as given below.")
cnn_perf = cnn_model.evaluate(x_test, y_test)
#cnn_perf = cnn_model.predict(x_test, y_test)

print("Transformer model performance is as given below.")
trans_perf = trans_model.evaluate(x_test, y_test)


  