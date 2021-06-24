import numpy as np
from PIL import Image

from utils import img2float, saturation_factor, contrast_factor, \
    img_pad, generate_corner_image, write_data_to_target, gen_crop_mask


def rotate_corner_bg(img_path, rot, height=300, width=300, scale=1.0, noise_bg=None, shift_x=None, shift_y=None,
                     saturation_f=None, contrast_f=None, get_target=True, rescale_dims=None, crop=False,
                     get_alpha=False, mask_only=False, merge_y=False):
    """
    Given a card image, generate a data point with the card
    :param img_path: Path of the image to include in the datapoint
    :param rot: Rotation of the card within the image
    :param height: Height of the generated image
    :param width: Width of the generated image
    :param scale: Scale of the card to be included
    :param noise_bg: Noise profile for the background of the image
    :param shift_x: Translation of the card along the x axis. The origin is at the center of the image
    :param shift_y: Translation of the card along the y axis. The origin is at the center of the image
    :param saturation_f: Saturation factor to apply to the card image
    :param contrast_f: Contrast factor to add to the card image
    :param get_target: Whether to return the target along with the input
    :param rescale_dims: Dimensions to rescale the inputs and targets **after** generation.
    :param crop: Whether to crop the card image in the input
    :param get_alpha: Whether to return the alpha channel with the input
    :param mask_only: Whether to return the partitions with the mask
    :param merge_y: Whether to merge the quad partitions into a top bottom partition.
    :return:
    """
    with open(img_path, "rb") as fp:


        if noise_bg is None:
            noise_bg = (0., 1., 0., 1., 0., 1.)

        image = Image.open(fp)

        # Generate a partition map for the target
        image_corners = generate_corner_image(height=204, width=146, merge_y=merge_y)

        # apply saturation factor
        if saturation_f is not None:
            image = saturation_factor(image, saturation_f)

        # Apply contrast factor
        if contrast_f is not None:
            image = contrast_factor(image, contrast_f)

        # Add alpha channel to card image
        image = image.convert("RGBA")

        mask_layer = None

        # If we crop the card image, we apply a crop mask to the card
        if crop:
            mask_layer = image.copy()
            mask = gen_crop_mask(image.width, image.height)
            img_data = np.array(image)
            img_data[..., 3] *= mask.astype("uint8")
            image = Image.fromarray(img_data)

        # We pad the input and target to allow rotation without loss of data
        image = img_pad(image, height, width)
        image_corners = img_pad(image_corners, height, width)

        if mask_layer is not None:
            mask_layer = img_pad(mask_layer, height, width)

        # We rotate the input and target
        image = image.rotate(-rot)
        image_corners = image_corners.rotate(-rot)

        if mask_layer is not None:
            mask_layer = mask_layer.rotate(-rot)

        # Define the center of the image
        tx = (image.width / 2) - (image.width / (2 * scale))
        ty = (image.height / 2) - (image.height / (2 * scale))

        # redefine the center of the image acording to given translation parameters
        if shift_x is not None and shift_y is not None:
            tx = ((image.width / 2) - (image.width / (2 * scale))) - (shift_x / scale)
            ty = (image.height / 2) - (image.height / (2 * scale)) - (shift_y / scale)
        elif shift_x is not None:
            tx = ((image.width / 2) - (image.width / (2 * scale))) - (shift_x / scale)
        elif shift_y is not None:
            ty = (image.height / 2) - (image.height / (2 * scale)) - (shift_y / scale)

        # Apply an affine transfor to the input and output to represent translation and scaling of the card
        image = image.transform(
            size=image.size,
            method=Image.AFFINE,
            data=(1 / scale, 0, tx, 0, 1 / scale, ty),
            resample=Image.NEAREST,
            fillcolor=(0, 0, 0, 0)
        )

        image_corners = image_corners.transform(
            size=image.size,
            method=Image.AFFINE,
            data=(1 / scale, 0, tx, 0, 1 / scale, ty),
            resample=Image.NEAREST,
            fillcolor=0
        )

        if mask_layer is not None:
            mask_layer = mask_layer.transform(
                size=image.size,
                method=Image.AFFINE,
                data=(1 / scale, 0, tx, 0, 1 / scale, ty),
                resample=Image.NEAREST,
                fillcolor=(0, 0, 0, 0)
            )

        # Image rescaling if necessary
        if rescale_dims is None:

            data = img2float(image)

            if mask_layer is not None:
                mask_data = img2float(mask_layer)[..., 3]

            data_target = np.zeros(shape=(height, width, 4 if merge_y else 6))
            data_target[..., 1:] = write_data_to_target(data_target[..., 1:], image_corners, merge_y=merge_y)

        else:

            image = image.resize(size=(int(width * rescale_dims), int(height * rescale_dims)))
            image_corners = image_corners.resize(
                size=(int(width * rescale_dims), int(height * rescale_dims)),
                resample=Image.NEAREST
            )

            if mask_layer is not None:
                mask_layer = mask_layer.resize(size=(int(width * rescale_dims), int(height * rescale_dims)))
                mask_data = np.array(img2float(mask_layer))[..., 3]

            data = img2float(image)

            data_target = np.zeros(shape=(int(height * rescale_dims), int(width * rescale_dims), 4 if merge_y else 6))
            data_target[..., 1:] = write_data_to_target(data_target[..., 1:], image_corners, merge_y=merge_y)

        if mask_layer is None:
            data_target[:, :, 0] = data.copy()[:, :, 3]
        else:
            data_target[:, :, 0] = mask_data

        data = noisify_bg_data(data, *noise_bg, get_alpha=get_alpha)

        if rescale_dims is None:
            assert data.shape == (height, width, 4 if get_alpha else 3)
            assert data_target.shape == (height, width, 6)
        else:
            assert data.shape == (int(height * rescale_dims), int(width * rescale_dims), 4 if get_alpha else 3)
            assert data_target.shape == (int(height * rescale_dims), int(width * rescale_dims), 6 if not merge_y else 4)

        assert data.dtype == "float64"

        # We return the data and target
        if get_target:
            if mask_only:
                return data, data_target[..., 0]
            else:
                return data, data_target
        else:
            return data


def noisify_bg_data(data, r_min=0., r_max=1., g_min=0., g_max=1., b_min=0., b_max=1., get_alpha=False):
    """
    Given an image with alpha, replace the alpha with a noisy RGB signal randomly generated from the specified RGB
    ranges
    :param data: Image to noisify
    :param get_alpha: Whether to erase the alpha channel after noise is added
    :return: Noisified image
    """
    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]
    a = data[:, :, 3]

    assert data.dtype == "float64", f'{data.dtype}'

    # We define the amount of noise for each channel independently
    noise_r = (r_max - r_min) * np.random.random_sample(size=r.shape) + r_min
    noise_g = (g_max - g_min) * np.random.random_sample(size=g.shape) + g_min
    noise_b = (b_max - b_min) * np.random.random_sample(size=b.shape) + b_min

    # Alpha blending of the noise with the original image
    noisy_r = (a * r) + ((1 - a) * noise_r)
    noisy_g = (a * g) + ((1 - a) * noise_g)
    noisy_b = (a * b) + ((1 - a) * noise_b)

    data[..., 0] = noisy_r
    data[..., 1] = noisy_g
    data[..., 2] = noisy_b

    if get_alpha:
        return data
    else:
        return data[..., :-1]
