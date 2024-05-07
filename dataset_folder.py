import tensorflow as tf
import os
from typing import List, Tuple, Callable, Optional

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def is_valid_image(filename: str, extensions: Tuple[str, ...] = IMG_EXTENSIONS) -> bool:
    return filename.lower().endswith(extensions)

def make_dataset(directory: str) -> List[Tuple[str, int]]:
    instances = []
    class_to_idx = {cls.name: idx for idx, cls in enumerate(sorted(os.scandir(directory), key=lambda x: x.name)) if cls.is_dir()}
    for class_name, idx in class_to_idx.items():
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        for root, _, fnames in os.walk(class_dir):
            for fname in sorted(fnames):
                if is_valid_image(fname):
                    path = os.path.join(root, fname)
                    instances.append((path, idx))
    return instances

def load_image(path: str) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize to [0, 1]
    return img

def prepare_for_training(ds, batch_size=32, shuffle_buffer_size=1000):
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()  # Repeat dataset indefinitely
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def create_dataset(root_dir: str, batch_size: int = 32) -> tf.data.Dataset:
    samples = make_dataset(root_dir)
    paths, labels = zip(*samples) if samples else ([], [])
    path_ds = tf.data.Dataset.from_tensor_slices(list(paths))
    label_ds = tf.data.Dataset.from_tensor_slices(list(labels))
    image_ds = path_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = tf.data.Dataset.zip((image_ds, label_ds))
    ds = prepare_for_training(ds, batch_size=batch_size)
    return ds

# Example usage
dataset = create_dataset('/path/to/dataset', batch_size=32)
