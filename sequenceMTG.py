from glob import glob

import numpy as np
import tensorflow as tf

from rotate import rotate_corner_bg
from utils import gen_rand_range, overlay_imgs


# Data generator for the TRS network
class MultiCornerSequenceMTG(tf.keras.utils.Sequence):

    def __init__(self, imgs, batch_size, img_size=(300, 300, 3), rescale_size=None, scale=True, shift=False,
                 saturation=False,
                 contrast=False, data_augment_factor=1.0, noise=None, empty_fraction=2, shift_range=50,
                 sigma=None, crop=False, crop_prob=1.0, max_cards=1, merge_y=False, crop_image=None):
        self.batch_size = batch_size
        self.image_size = img_size
        self.scale = scale
        self.augment = 1 if int(data_augment_factor) <= 1 else int(data_augment_factor)
        self.images = imgs
        self.npaths = len(imgs)
        self.dims = (batch_size,) + img_size
        self.noise = noise
        self.empty = empty_fraction
        self.shift = shift
        self.saturation = saturation
        self.contrast = contrast
        self.shift_range = shift_range
        self.rescale_size = rescale_size
        self.sigma = sigma
        self.crop = crop
        self.max_cards = max_cards

        self.resize_dims = (
                (batch_size,) +
                (int(img_size[0] * rescale_size), int(img_size[1] * rescale_size)) +
                (img_size[-1],)
        ) if self.rescale_size is not None else None

        self.merge_y = merge_y
        self.crop_image = crop_image
        self.crop_prob = crop_prob

    def __getitem__(self, item):

        # Data numbers = image_number * augment + augment number
        start_index = (item * self.batch_size)

        if self.max_cards <= 1:
            max_range = 1
        else:
            max_range = np.random.randint(1, self.max_cards + 1)

        # if resize dimentions are defined, we resize the input image to the given dimensions
        if self.resize_dims is not None:
            data = np.zeros(shape=self.resize_dims)
            target = np.zeros(shape=self.resize_dims[:-1] + (6 if not self.merge_y else 4,))
        else:
            data = np.zeros(shape=self.dims)
            target = np.zeros(shape=self.dims[:-1] + (6 if not self.merge_y else 4,))

        # For each datapoint in the batch
        for i in range(self.batch_size):

            # get the index of the image for this datapoint
            img_index = (i + start_index) // self.augment

            images = []
            targets = []

            scale = np.random.uniform(0.8, 1.2) if self.scale else 1.0
            shift_x = np.random.normal(0.0, self.shift_range / 2) if self.shift else None
            shift_y = np.random.normal(0.0, self.shift_range / 2) if self.shift else None
            contrast_f = 2.0 ** np.random.uniform(-1, 0.2) if self.contrast else None
            saturation_f = 2.0 ** np.random.uniform(-1, 0.2) if self.saturation else None

            # Generate an image and its target
            inpt, oupt = rotate_corner_bg(
                self.images[img_index],
                rot=np.random.randint(low=0, high=360),
                height=self.dims[1],
                width=self.dims[2],
                scale=scale,
                noise_bg=gen_rand_range(),
                shift_x=shift_x,
                shift_y=shift_y,
                saturation_f=saturation_f,
                contrast_f=contrast_f,
                rescale_dims=self.rescale_size,
                crop=(self.crop and max_range == 1),
                get_alpha=True,
                merge_y=self.merge_y

            )

            images.append(inpt)
            targets.append(oupt)

            # If there is more than one card in the image, we insert those to the list
            for l in range(1, max_range):
                shift_x = np.random.normal(0.0, self.shift_range / 2) if self.shift else None
                shift_y = np.random.normal(0.0, self.shift_range / 2) if self.shift else None
                contrast_f = np.random.uniform(1.0, 1.5) if self.contrast else None
                saturation_f = np.random.uniform(1.0, 1.5) if self.saturation else None

                inpt, oupt = rotate_corner_bg(
                    self.images[np.random.randint(0, len(self.images))],
                    np.random.randint(low=0, high=360),
                    self.dims[1],
                    self.dims[2],
                    scale=scale,
                    noise_bg=gen_rand_range(),
                    shift_x=shift_x,
                    shift_y=shift_y,
                    saturation_f=saturation_f,
                    contrast_f=contrast_f,
                    rescale_dims=self.rescale_size,
                    crop=self.crop,
                    get_alpha=True,
                    merge_y=self.merge_y
                )
                images.append(inpt)
                targets.append(oupt)

            data[i], target[i] = overlay_imgs(images, targets)

        # If we do not crop the data and targets, we return them as is
        if self.crop_image is None:
            return (
                data,
                (
                    target[..., 0],
                    target[..., 1:]
                )
            )
        else:   # We return a cropped input and output by removing an amount of pixels from each side of the image
            return (
                data[..., self.crop_image:-self.crop_image, self.crop_image:-self.crop_image, :],
                (
                    target[..., self.crop_image:-self.crop_image, self.crop_image:-self.crop_image, 0],
                    target[..., self.crop_image:-self.crop_image, self.crop_image:-self.crop_image, 1:]
                )
            )

    def __len__(self):
        return int((self.npaths * self.augment) / self.batch_size)

    # Factory method to get the training generator and validation generator together
    @classmethod
    def get_generators(cls, img_dir, train_test_split=0.9, batch_size=50, img_size=(300, 300, 3), shuffle=True,
                       scale=True, shift=False, saturation=False, contrast=False, data_augment_factor=1.0,
                       max_data_points=None, shift_range=50, empty_fraction=2, rescale_dims=None, sigma=None,
                       crop=False, crop_prob=1.0, max_cards=1, merge_y=False, crop_image=None):
        if max_data_points is None:
            images = glob(f'{img_dir}/*.jpg')
            print(len(images))
        else:
            images = glob(f'{img_dir}/*.jpg')[:max_data_points]

        if shuffle:
            np.random.shuffle(images)

        split_index = int(len(images) * train_test_split)
        train_set = images[:split_index]
        test_set = images[split_index:]

        train_generator = cls(
            train_set,
            batch_size=batch_size,
            img_size=img_size,
            scale=scale,
            shift=shift,
            saturation=saturation,
            contrast=contrast,
            data_augment_factor=data_augment_factor,
            shift_range=shift_range,
            empty_fraction=empty_fraction,
            rescale_size=rescale_dims,
            sigma=sigma,
            crop=crop,
            max_cards=max_cards,
            merge_y=merge_y,
            crop_image=crop_image,
            crop_prob=crop_prob

        )

        test_generator = cls(
            test_set,
            batch_size=batch_size,
            img_size=img_size,
            scale=scale,
            shift=shift,
            saturation=saturation,
            contrast=contrast,
            data_augment_factor=data_augment_factor,
            shift_range=shift_range,
            empty_fraction=empty_fraction,
            rescale_size=rescale_dims,
            sigma=sigma,
            crop=crop,
            max_cards=max_cards,
            merge_y=merge_y,
            crop_image=crop_image,
            crop_prob=crop_prob

        )

        return train_generator, test_generator
