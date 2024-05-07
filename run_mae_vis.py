import argparse
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

def get_args():
    parser = argparse.ArgumentParser(description='MAE visualization reconstruction script')
    parser.add_argument('img_path', type=str, help='Input image path')
    parser.add_argument('save_path', type=str, help='Save image path')
    parser.add_argument('model_path', type=str, help='Checkpoint path of model')
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='ratio of the visual tokens/patches need be masked')
    return parser.parse_args()

def load_image(img_path, size):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((size, size))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def save_image(img_array, save_path):
    img = array_to_img(img_array[0])
    img.save(save_path)

def reconstruct_image(model, img, mask_ratio, save_path, img_original_path):
    input_tensor = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0
    mask = np.random.rand(*input_tensor.shape[1:3]) < mask_ratio
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_img = input_tensor.numpy()
    masked_img[mask] = 0

    prediction = model(masked_img)
    prediction[mask] = masked_img[mask]  # Set the unmasked region to original

    save_image(input_tensor * 255.0, img_original_path)
    save_image(prediction * 255.0, save_path)

def main(args):
    model = load_model(args.model_path)
    img = load_image(args.img_path, args.input_size)

    reconstruct_image(
        model=model,
        img=img,
        mask_ratio=args.mask_ratio,
        save_path=os.path.join(args.save_path, 'rec_img.jpg'),
        img_original_path=os.path.join(args.save_path, 'ori_img.jpg')
    )

if __name__ == '__main__':
    args = get_args()
    main(args)
