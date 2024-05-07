import tensorflow as tf
from PIL import Image
import random
import math
import numpy as np

def _tf_interp(method):
    if method == 'bicubic':
        return tf.image.ResizeMethod.BICUBIC
    elif method == 'lanczos':
        return tf.image.ResizeMethod.LANCZOS3
    elif method == 'hamming':
        return tf.image.ResizeMethod.HAMMING
    else:
        return tf.image.ResizeMethod.BILINEAR

class RandomResizedCropAndInterpolationWithTwoPic:
    def __init__(self, size, second_size=None, scale=(0.08, 1.0), ratio=(3./4., 4./3.),
                 interpolation='bilinear', second_interpolation='lanczos'):
        self.size = size
        self.second_size = second_size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = _tf_interp(interpolation)
        self.second_interpolation = _tf_interp(second_interpolation) if second_size is not None else None

    def __call__(self, img):
        img = tf.convert_to_tensor(img)
        img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float

        # Calculate the aspect ratio and area bounds
        img_shape = tf.shape(img)[:2]
        area = tf.cast(img_shape[0] * img_shape[1], tf.float32)

        for _ in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img_shape[1] and h <= img_shape[0]:
                i = random.randint(0, img_shape[0] - h)
                j = random.randint(0, img_shape[1] - w)
                crop_img = tf.image.crop_to_bounding_box(img, i, j, h, w)
                resized_img = tf.image.resize(crop_img, [self.size, self.size], method=self.interpolation)
                if self.second_size:
                    second_resized_img = tf.image.resize(crop_img, [self.second_size, self.second_size], method=self.second_interpolation)
                    return resized_img, second_resized_img
                return resized_img

        # Fallback to a central crop
        in_ratio = img_shape[1] / img_shape[0]
        if in_ratio < min(self.ratio):
            w = img_shape[1]
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = img_shape[0]
            w = int(round(h * max(self.ratio)))
        else:
            w = img_shape[1]
            h = img_shape[0]
        i = (img_shape[0] - h) // 2
        j = (img_shape[1] - w) // 2
        crop_img = tf.image.crop_to_bounding_box(img, i, j, h, w)
        resized_img = tf.image.resize(crop_img, [self.size, self.size], method=self.interpolation)
        if self.second_size:
            second_resized_img = tf.image.resize(crop_img, [self.second_size, self.second_size], method=self.second_interpolation)
            return resized_img, second_resized_img
        return resized_img

    def __repr__(self):
        interpolate_str = self.interpolation
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0}'.format(interpolate_str)
        if self.second_size is not None:
            format_string += ', second_size={0}'.format(self.second_size)
            format_string += ', second_interpolation={0}'.format(self.second_interpolation)
        format_string += ')'
        return format_string

# Example usage
transform = RandomResizedCropAndInterpolationWithTwoPic(size=224, second_size=256, scale=(0.5, 1.0), ratio=(3./4., 4./3.))
img = Image.open('path_to_image.jpg')
img_tensor = np.array(img)
transformed_img = transform(img_tensor)
