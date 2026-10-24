import tensorflow as tf
from tensorflow.keras import layers, Model


@tf.keras.utils.register_keras_serializable(package="SRGAN")
class PixelShuffle(layers.Layer):

    def __init__(self, scale=2, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


def residual_block(x, filters=64):

    shortcut = x

    y = layers.Conv2D(filters, 3, padding="same")(x)
    y = layers.BatchNormalization()(y)
    y = layers.PReLU(shared_axes=[1, 2])(y)

    y = layers.Conv2D(filters, 3, padding="same")(y)
    y = layers.BatchNormalization()(y)

    return layers.Add()([shortcut, y])


def upsample_block(x, filters=256):

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = PixelShuffle(scale=2)(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    return x


def build_generator(num_residual_blocks=8):

    inputs = layers.Input(shape=(32, 32, 3))

    x = layers.Conv2D(64, 9, padding="same")(inputs)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    long_skip = x

    for _ in range(num_residual_blocks):
        x = residual_block(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, long_skip])

    x = upsample_block(x)
    x = upsample_block(x)

    outputs = layers.Conv2D(
        3, 9, padding="same", activation="sigmoid"
    )(x)

    return Model(inputs, outputs, name="SRGAN_Generator")


def discriminator_block(x, filters, strides=1, use_bn=True):

    x = layers.Conv2D(
        filters, 3, strides=strides, padding="same"
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x)

    return layers.LeakyReLU(negative_slope=0.2)(x)


def build_discriminator():

    inputs = layers.Input(shape=(128, 128, 3))

    x = discriminator_block(inputs, 64, 1, False)
    x = discriminator_block(x, 64, 2)
    x = discriminator_block(x, 128, 1)
    x = discriminator_block(x, 128, 2)
    x = discriminator_block(x, 256, 1)
    x = discriminator_block(x, 256, 2)
    x = discriminator_block(x, 512, 1)
    x = discriminator_block(x, 512, 2)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    return Model(
        inputs, outputs,
        name="SRGAN_Efficient_Discriminator"
    )
