import tensorflow as tf
import numpy as np
import json
import os
import datetime
import argparse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from datasets import build_dataset
from models import create_model  # Define your model creation function
from utils import setup_logger, save_model

def get_args():
    parser = argparse.ArgumentParser('MAE fine-tuning and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    # Add other arguments
    return parser.parse_args()

def main(args):
    # Set up the logging and saving directories
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logger = setup_logger(output=args.output_dir, name="image_classification")

    # Data loading
    train_dataset, num_classes = build_dataset(is_train=True, args=args)
    val_dataset, _ = build_dataset(is_train=False, args=args)

    # Build model
    model = create_model(num_classes=num_classes, input_size=(args.input_size, args.input_size, 3))
    model.compile(optimizer=Adam(learning_rate=args.lr),
                  loss=CategoricalCrossentropy(label_smoothing=args.smoothing),
                  metrics=[CategoricalAccuracy()])

    # Callbacks for saving, logging, and learning rate schedule
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.output_dir, 'ckpt_{epoch}'), save_weights_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=args.log_dir)
    ]

    # Train the model
    history = model.fit(train_dataset, epochs=args.epochs, callbacks=callbacks, validation_data=val_dataset)

    # Save the model at the end of training
    save_model(model, os.path.join(args.output_dir, 'final_model'))

    # Optionally log training progress
    logger.info("Training completed. Model saved to {}".format(args.output_dir))

if __name__ == '__main__':
    args = get_args()
    main(args)
