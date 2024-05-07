import tensorflow as tf
import argparse
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from utils import build_pretraining_dataset, create_model
from training import train_one_epoch  # Assume this handles training logic

def get_args():
    parser = argparse.ArgumentParser('MAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str)
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--drop_path', type=float, default=0.0)
    parser.add_argument('--normlize_target', default=True, type=bool)
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--data_path', default='/path/to/dataset', type=str)
    parser.add_argument('--output_dir', default='./training_output', type=str)
    parser.add_argument('--log_dir', default='./logs', type=str)
    parser.add_argument('--device', default='gpu', type=str)
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args()

def main(args):
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    # Set the seed for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Prepare datasets
    train_dataset = build_pretraining_dataset(args.data_path, args.batch_size, args.mask_ratio)

    # Build model
    model = create_model(args.model, input_size=args.input_size, mask_ratio=args.mask_ratio)

    # Setup optimizer
    if args.opt.lower() == 'adamw':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    else:
        raise ValueError('Optimizer not supported!')

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

    # Prepare callbacks for saving the model and logging
    callbacks = [
        ModelCheckpoint(filepath=os.path.join(args.output_dir, 'ckpt_{epoch}'), save_weights_only=True, save_freq=args.save_ckpt_freq),
        TensorBoard(log_dir=args.log_dir)
    ]

    # Start training
    model.fit(train_dataset, epochs=args.epochs, callbacks=callbacks)

    # Save final model
    model.save(os.path.join(args.output_dir, 'final_model'))

if __name__ == '__main__':
    args = get_args()
    main(args)
