
import tensorflow as tf

SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, tf.cast(label, tf.float32)


def create_dataset(dataframe, training=False):

    paths = dataframe["filepath"].astype(str).values
    labels = dataframe["label"].astype("float32").values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    if training:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomContrast(0.1)
        ])

        dataset = dataset.map(
            lambda images, labels: (
                tf.clip_by_value(
                    augmentation(images, training=True),
                    0.0,
                    1.0
                ),
                labels
            ),
            num_parallel_calls=AUTOTUNE
        )

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset
