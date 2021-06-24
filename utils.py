import numpy as np
import tensorflow as tf
from PIL import ImageEnhance, Image, ImageOps
from scipy import stats as st
from scipy.ndimage import gaussian_filter


def build_model(input_shape, layers, name=None) -> tf.keras.Model:
    """
    Take an input shape and a list of layers and build a sequential model with the given input shape and layers.
    :param input_shape: Input shape of the model
    :param layers: layers of the model
    :param name: Name of the model
    :return: Built and instanced model
    """
    if isinstance(input_shape, list):
        inpt = []
        for i in input_shape:
            inpt.append(tf.keras.Input(shape=i))
    else:
        inpt = tf.keras.Input(shape=input_shape)

    x = inpt
    for l in layers:
        x = l(x)

    return tf.keras.Model(inpt, x, name=name)


def img_pad(img, height, width):
    """
    Take a PIL image and pad it to the specified width and height
    :param img: PIL image to pad
    :param height: Height of the padded image
    :param width: Width of the padded image
    :return: Padded image
    """
    old_size = img.size
    delta_w = width - old_size[0]
    delta_h = height - old_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    return ImageOps.expand(img, padding)


def write_data_to_target(data, image: Image, merge_y=False):
    """
    Take a partition image where each quadrant of the image has pixels with a different value and writes it to a tensor
    where each part of the last dimension of the tensor is a quadrant
    :param data: Tensor in which to write the partition
    :param image: Partition image
    :param merge_y: If merge_y is true then the partition are binary (top-bottom) else the partitions are quadrants
    (top-left, top-right, bottom-left, bottom-right)
    :return:
    """
    if merge_y:    # if the partitions are top-bottom
        assert data.shape[-1] == 3  # There are only three classes (top, bottom, neither
        img_data = np.array(image)
        data[..., 2] = np.where(img_data == 0, 1, 0)    # We fill in the outside class first

        for i in range(1, 3):   # For every partition, we fill in in its own slice of the tensor
            data[..., i - 1] = np.where(img_data == i, 1, 0)

    else:   # Same as with merge_y except there are five classes
        assert data.shape[-1] == 5
        img_data = np.array(image)

        data[..., 4] = np.where(img_data == 0, 1, 0)

        for i in range(1, 5):
            data[..., i - 1] = np.where(img_data == i, 1, 0)

    return data


def generate_corner_image(height, width, merge_y=False) -> Image:
    """
    Generate a partition image where eah partition has pixels of a given value
    :param height: Height of the partition image
    :param width: Width of the partition image
    :param merge_y: Whether to use 3 or 5 classes for the partitioning
    :return: Partition image
    """
    array = np.zeros(shape=(height, width))
    h = height // 2
    w = width // 2

    if merge_y:
        array[:h, :] = 1
        array[h:, :] = 2
    else:
        array[:h, :w] = 1
        array[:h, w:] = 2
        array[h:, :w] = 3
        array[h:, w:] = 4

    return Image.fromarray(array)

def overlay_imgs(images, targets):
    """
        Takes a series of input images and output targets and merges them into one input and one target.
        Used for card overlay
        :param images: Input images to overlay
        :param targets: Output targets to overlay. If None, only the inputs are merged and returned
        :return: Merged input, target pair for a data point containing multiple cards
        """
    data = np.array(images[0].copy()[..., :-1])

    for i in range(1, len(images)):
        image = images[i][..., :-1]
        mask = np.repeat(images[i][..., -1, np.newaxis], 3, axis=-1)

        data = ((1 - mask) * data) + (mask * image)

    if targets is not None:

        data_target = targets[-1]

        data_target[data_target > 1] = 1

        return data, data_target
    else:
        return data


def onehot(i, length):
    """
    Generate onehot vector of of given length and with (i+1)th value set to 1
    """
    if i >= length:
        raise ValueError

    oh = np.zeros((length,)).astype("float32")
    oh[i] = 1

    return oh


def img2float(img):
    """
    Take a given image (ndarray or pil image) and return a float64 numpy array with pixel values normalized between
    0 and 1
    :param img: Image to convert
    :return: Converted image
    """
    x = np.array(img).astype("float64")

    if np.max(x) > 1.0:
        return x / 255.0
    else:
        return x


def gen_rand_range():
    """
    Generate three random ranges between 0 and 1 used for random background generation
    :return: Vector of 6 values (min_r, max_r, min_g, max_g, min_b, max_b),
    """
    range_r = np.random.random_sample(size=(2,))
    range_g = np.random.random_sample(size=(2,))
    range_b = np.random.random_sample(size=(2,))

    range_r.sort()
    range_g.sort()
    range_b.sort()

    assert range_r[0] < range_r[1]
    assert range_g[0] < range_g[1]
    assert range_b[0] < range_b[1]

    return np.concatenate((range_r, range_g, range_b))


def saturation_factor(img, factor):
    """
    Add a saturation filter to an image (see ImageEnhance.Color doc).
    :param img: Image to saturate
    :param factor: Factor by which to saturate. 0 is black and white. 1 is the original.
    Factor > 1 augments the pixel intensities by that factor
    :return: saturated or desaturated image.
    """
    c = ImageEnhance.Color(img)
    c = c.enhance(factor)
    return c


def contrast_factor(img, factor):
    """
    Add a contrast filter to an image (see ImageEnhance.Contrast).
    :param img: imae to contrast
    :param factor: Factor by which to augment or reduce contrast of the image. 0 gives a uniform image. 1 gives the
    original image. As factor -> inf, the image becomes binarized.
    :return: Contrasted image
    """
    c = ImageEnhance.Contrast(img)
    c = c.enhance(factor)
    return c

def gen_crop_mask(w, h, crop_prob=1.0):
    """
    Generates a mask which represents if a pixel is cropped or not. The function defines a random line and every pixel
    under that line is cropped
    :param w: Original width of the uncropped mask
    :param h: Original Height of the uncropped mask
    :param crop_prob: Value between 0 and 1. Probability that the card will be cropped. If the card isn't cropped,
    the mask is simply empty
    :return: Mask for the cropping using a randomly drawn line
    """

    # We roll randomly to see if the card is cropped. If it isn't, we give the empty mask.
    if np.random.random() > crop_prob:
        return np.zeros(shape=(w, h))

    # Pixel (x,y) will be hidden if ax + by + c >= 0 for a random a, b and c

    mx = np.array([[i for i in range(w)] for j in range(h)]) - (w / 2)
    my = np.array([[j for i in range(w)] for j in range(h)]) - (h / 2)

    g = np.random.uniform(0, 2 * np.pi)
    a = np.sin(g)
    b = np.cos(g)

    s = np.sqrt(w ** 2 + h ** 2) / 2

    c = np.random.uniform(0, s)

    mask = (a * mx) + (b * my) + c  # ax + by + c

    mask = (mask >= 0).astype("float")

    return mask


def trs(mask, top, bottom):
    """
    Given a card mask and its top-bottom partition, find the translation, rotation and scale (TRS) of the given card.
    """
    w, h = mask.shape

    mx = np.array([[i for i in range(w)] for j in range(h)])
    my = np.array([[j for i in range(w)] for j in range(h)])

    # The center of the mask is its centroid or average.
    x_center = np.sum(mx * mask) / np.sum(mask)
    y_center = np.sum(my * mask) / np.sum(mask)

    # Find the centroid of the top and bottom partitions respectivaly.
    x_top = np.sum(mx * top) / np.sum(top)
    y_top = np.sum(my * top) / np.sum(top)

    x_bot = np.sum(mx * bottom) / np.sum(bottom)
    y_bot = np.sum(my * bottom) / np.sum(bottom)

    # Find the rotation vector using the centroid of the top and bottom partitions
    v = np.array([x_top - x_bot, y_top - y_bot])
    v /= np.linalg.norm(v)

    # Find the angle of the card using atan2 on the rotation vector
    theta = np.arctan2(-v[1], v[0])

    # Find the scale of the card by doing a ration of the mask pixels over the mask pixels of a "normal" card (204x146).
    scale = np.sqrt(np.sum(mask) / ((204 * 146) / 4))

    return (
        np.array([x_center, y_center]),
        np.array([x_top, y_top]),
        np.array([x_bot, y_bot]),
        theta * (180 / np.pi),
        scale
    )


def unit_step(x):
    return int(x >= 0)


def nms_kernel(v, t):
    """
    Apply a non-maximum suppression kernel of a given window.
    :param v: Sug-image on which to apply nms
    :param t: nms threshold
    :return: NSM value for the given window and threshold
    """
    assert len(v.shape) in [2, 3]

    if len(v.shape) == 3:
        assert v.shape[2] == 1
        v = v.reshape(v.shape[:-1])

    assert v.shape[0] == v.shape[1]
    assert v.shape[0] % 2 == 1

    c = v.shape[0] // 2

    val = unit_step(np.min(v[c, c] - v)) * (unit_step(v[c, c] - t))

    return val


def nms(image, r, t=0.5, padding=False):
    """
    Non-maximum suppression for card localisation within an image
    :param image: Image on which to apply NMS
    :param r: Radius of the kernel used for nms
    :param t: Threshold for the NMS algorithm
    :param padding: Padding to insert around the image in order to prevent problems at the edge of the image.
    :return:
    """
    original_shape = image.shape[:-1]

    image = np.squeeze(image)

    if padding:
        image = np.pad(image, (r, r,))

    oupt = np.zeros(shape=image.shape)
    shape = image.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            if (i - r > 0) and (j - r > 0) and (i + r < shape[0]) and (j + r < shape[1]):
                oupt[i, j] = nms_kernel(image[i - r:i + r + 1, j - r:j + r + 1], t)
            else:
                oupt[i, j] = 0.0

    if padding:
        oupt = oupt[r:-r, r:-r]

    assert oupt.shape == original_shape

    return oupt
