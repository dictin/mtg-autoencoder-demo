import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import argparse
import json
import tensorflow as tf
from sequenceMTG import MultiCornerSequenceMTG
import time
from utils import build_model


class AutoencoderMTG(tf.keras.Model):

    def __init__(self, width, height, depth, filters=(16, 8), latent_dim=16, chan_dim=-1, kernel=(3, 3), alpha=0.2,
                 name=None, crop_mask=None, crop_corners=None):
        """
        :param width: Width of the network input
        :param height: Height of the network input
        :param depth: Depth of the network input
        :param filters: Tuple of the number of feature maps for each layer
        :param latent_dim: 1D dimension of the latent space of the autoencoder
        :param chan_dim: Which dimension of the data to consider as the color channels
        :param kernel: default CNN kernel size
        :param alpha: Alpha to use for LeakyRelu activations (legacy code, not used)
        :param name: Model name
        :param crop_mask: Size of the cropping layer for the mask branch
        :param crop_corners: Size of the crop layer for the corner branch
        """
        super(AutoencoderMTG, self).__init__(name=name)
        self.input_dim = (height, width, depth)
        self.chan_dim = chan_dim
        self.layers_encode = []  # Encoder layers
        self.layers_decode_1 = []  # Decoder layers for mask branch
        self.layers_decode_2 = []  # Decoder layers for the corner branch
        self.alpha = alpha

        # Building the encoder

        self.layers_encode.append(
            tf.keras.layers.Conv2D(
                input_shape=self.input_dim,
                filters=filters[0],
                kernel_size=kernel,
                strides=(2, 2),
                padding="same",
                bias_initializer=tf.keras.initializers.Zeros()
            )
        )
        self.layers_encode.append(
            tf.keras.layers.ReLU()
        )

        for f in filters[1:]:
            self.layers_encode.append(
                tf.keras.layers.Conv2D(
                    filters=f,
                    kernel_size=kernel,
                    strides=(2, 2),
                    padding="same",
                    bias_initializer=tf.keras.initializers.Zeros()
                )
            )
            self.layers_encode.append(
                tf.keras.layers.ReLU()
            )

        self.layers_encode.append(
            tf.keras.layers.Conv2D(
                filters=latent_dim,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",
                name="latent_space",
                bias_initializer=tf.keras.initializers.Zeros()
            )
        )

        # Building the mask branch

        self.layers_decode_1.append(
            tf.keras.layers.ReLU()
        )

        self.layers_decode_1.append(
            tf.keras.layers.Conv2DTranspose(
                filters=256,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",
                bias_initializer=tf.keras.initializers.Zeros()
            )
        )

        self.layers_decode_1.append(
            tf.keras.layers.ReLU()
        )

        for f in filters[::-1]:
            self.layers_decode_1.append(
                tf.keras.layers.Conv2DTranspose(
                    filters=f,
                    kernel_size=kernel,
                    strides=(2, 2),
                    padding="same",
                    bias_initializer=tf.keras.initializers.Zeros()
                )
            )

            self.layers_decode_1.append(
                tf.keras.layers.ReLU()
            )

        self.layers_decode_1.append(
            tf.keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=(3, 3),
                padding="valid",
                bias_initializer=tf.keras.initializers.Zeros()
            )
        )

        self.layers_decode_1.append(
            tf.keras.layers.Cropping2D(cropping=crop_mask)
        )

        self.layers_decode_1.append(
            tf.keras.layers.Activation(activation="sigmoid", name="mask")
        )

        # Building the corner branch

        for s in [6, 8, 10, 12]:
            self.layers_decode_2.append(
                tf.keras.layers.Conv2DTranspose(
                    filters=s,
                    kernel_size=(3, 3),
                    padding="valid",
                    bias_initializer=tf.keras.initializers.Zeros()
                )
            )

            self.layers_decode_2.append(
                tf.keras.layers.ReLU()
            )

        self.layers_decode_2.append(
            tf.keras.layers.Conv2DTranspose(
                filters=3,
                kernel_size=(3, 3),
                padding="valid",
                bias_initializer=tf.keras.initializers.Zeros()
            )
        )

        self.layers_decode_2.append(
            tf.keras.layers.Cropping2D(cropping=crop_corners)
        )

        self.layers_decode_2.append(
            tf.keras.layers.Softmax(name="corners", axis=-1)
        )

        # Building sub-networks
        encode_model = build_model((height, width, depth), self.layers_encode, name="encode_model")
        recurrent_hook = build_model((1, 1, 1024), self.layers_decode_1[:-3], name="recurrent_hook")
        decode_mask = build_model(recurrent_hook.output_shape[1:], self.layers_decode_1[-3:], name="decode_mask")
        decode_corners = build_model(recurrent_hook.output_shape[1:], self.layers_decode_2, name="decode_corners")

        # Building the full network

        inputs = tf.keras.Input(shape=(height, width, depth))

        encode_outputs = encode_model(inputs)
        hook_output = recurrent_hook(encode_outputs)
        mask_outputs = decode_mask(hook_output)
        corners_outputs = decode_corners(hook_output)

        self.autoencoder = tf.keras.Model(inputs, [mask_outputs, corners_outputs])

        encode_model.summary()
        recurrent_hook.summary()
        decode_mask.summary()
        decode_corners.summary()

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.autoencoder(x)

        return x


def get_model(args):

    if args.retrain is None:
        print("loaded new model")
        autoencoder = AutoencoderMTG(
            height=150,
            width=150,
            depth=3,
            filters=(4, 8, 16, 32, 64, 128),
            latent_dim=1024,
            kernel=(3, 3),
            alpha=0.5,
            name="AutoMTG_mask_1",
            crop_mask=(22, 22),
            crop_corners=(26, 26)
        )

        return autoencoder.autoencoder
    else:
        print("retraining old model")
        return tf.keras.models.load_model(args.retrain)


if __name__ == '__main__':

    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
        exit()

    parser = argparse.ArgumentParser()

    parser.add_argument("epochs", nargs='?', default=-1)
    parser.add_argument("--config", default=None)
    parser.add_argument("--retrain", default=None)
    parser.add_argument("--batch_size", default=16)

    args = parser.parse_args()

    epochs = int(args.epochs)
    no_ui = bool(args.no_ui)
    batch_size = int(args.batch_size)

    if args.config:
        with open(args.config, 'r') as fp:
            paths = json.load(fp)
            model_save = paths["model_save"]
            log_file = paths["log_file"]
            graph_file = paths["graph_file"]
            dataset_dir = paths["dataset_dir"]
    else:
        model_save = f'tf_models/model_{time.time()}_autoencoder_corners.tf'
        log_file = f'logs/log_{time.time()}_autoencoder_corners.log'
        graph_file = f'logs/learning_graphs/log_{time.time()}_autoencoder_corners.log'
        dataset_dir = "mtg_img"

    print(model_save)

    model = get_model(args)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=model_save, save_best_only=True, monitor="val_loss", verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=100),
        tf.keras.callbacks.TensorBoard(log_dir="tensorboard/logs", histogram_freq=1),
    ]

    train_set, test_set = MultiCornerSequenceMTG.get_generators(
        img_dir=dataset_dir,
        img_size=(300, 300, 3),
        batch_size=batch_size,
        data_augment_factor=1.0,
        train_test_split=0.8,
        scale=True,
        shift=True,
        contrast=True,
        saturation=True,
        shift_range=75,
        rescale_dims=0.5,
        sigma=3,
        crop=False,
        max_cards=2,
        max_data_points=20000,
        merge_y=True
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            lr=1e-4,
        ),
        loss={
            "decode_mask": tf.keras.losses.BinaryCrossentropy(),
            "decode_corners": tf.keras.losses.CategoricalCrossentropy()
        },
        loss_weights=[1, 1],
    )

    if epochs < 0:
        model.summary()
        exit(0)

    model.fit(
        x=train_set,
        epochs=epochs,
        validation_data=test_set,
        shuffle=False,
        callbacks=callbacks
    )

    exit(0)
