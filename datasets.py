import tensorflow as tf
import os

# Constants for normalization, adjust these based on your actual requirements
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_INCEPTION_MEAN = [0.5, 0.5, 0.5]
IMAGENET_INCEPTION_STD = [0.5, 0.5, 0.5]

class DataAugmentationForMAE:
    def __init__(self, args):
        self.input_size = args.input_size
        self.mean = IMAGENET_DEFAULT_MEAN if args.imagenet_default_mean_and_std else IMAGENET_INCEPTION_MEAN
        self.std = IMAGENET_DEFAULT_STD if args.imagenet_default_mean_and_std else IMAGENET_INCEPTION_STD
        self.mask_ratio = args.mask_ratio

    def __call__(self, image):
        # Randomly resize and crop the image
        image = tf.image.random_crop(image, size=[self.input_size, self.input_size, 3])
        image = tf.image.resize(image, [self.input_size, self.input_size])
        # Normalize the image
        image = (image - self.mean) / self.std
        # Apply random masking
        mask = tf.random.uniform(shape=tf.shape(image)[:2], minval=0, maxval=1)
        mask = tf.cast(mask < self.mask_ratio, tf.float32)
        return image * mask[:, :, tf.newaxis]

    def __repr__(self):
        return f"(DataAugmentationForMAE, transform = RandomResizedCrop, Normalize, MaskRatio={self.mask_ratio})"

def load_and_preprocess_image(file_path, args):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [args.input_size, args.input_size])
    img = (img / 255.0 - args.mean) / args.std  # Normalize
    return img

def build_dataset(is_train, args):
    if args.data_set == 'CIFAR':
        dataset = tf.keras.datasets.cifar100.load_data()[0 if is_train else 1]
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = tf.data.Dataset.list_files(root + '/*/*.jpg')
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = tf.data.Dataset.list_files(root + '/*/*.jpg')
        nb_classes = args.nb_classes
    else:
        raise NotImplementedError()

    # Load and preprocess images
    dataset = dataset.map(lambda x: load_and_preprocess_image(x, args))

    return dataset, nb_classes
