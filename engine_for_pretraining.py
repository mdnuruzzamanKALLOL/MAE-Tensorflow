import tensorflow as tf
import math
import sys
from typing import Iterable

import utils  # Ensure utils is adapted for TensorFlow use
from einops import rearrange
from tensorflow.keras import losses

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

class TrainOneEpoch:
    def __init__(self, model, optimizer, device, loss_scale_manager, max_norm, patch_size, normalize_target, log_writer=None, lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_scale_manager = loss_scale_manager
        self.max_norm = max_norm
        self.patch_size = patch_size
        self.normalize_target = normalize_target
        self.log_writer = log_writer
        self.lr_scheduler = lr_scheduler
        self.metric_logger = utils.MetricLogger(delimiter="  ")
        self.loss_func = losses.MeanSquaredError()

    def train_one_epoch(self, data_loader, epoch, start_steps, lr_schedule_values, wd_schedule_values):
        self.model.train()
        self.metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        self.metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = f'Epoch: [{epoch}]'
        print_freq = 10

        for step, (images, bool_masked_pos) in enumerate(self.metric_logger.log_every(data_loader, print_freq, header)):
            it = start_steps + step  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None:
                self._adjust_learning_rate_and_weight_decay(it, lr_schedule_values, wd_schedule_values)

            with tf.GradientTape() as tape:
                images = tf.convert_to_tensor(images, dtype=tf.float32)
                bool_masked_pos = tf.reshape(tf.convert_to_tensor(bool_masked_pos, dtype=tf.bool), shape=[-1])

                # Normalize and patch the images
                mean = tf.constant(IMAGENET_DEFAULT_MEAN, shape=[1, 1, 1, 3], dtype=tf.float32)
                std = tf.constant(IMAGENET_DEFAULT_STD, shape=[1, 1, 1, 3], dtype=tf.float32)
                unnorm_images = images * std + mean

                if self.normalize_target:
                    images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=self.patch_size, p2=self.patch_size)
                    images_norm = (images_squeeze - tf.reduce_mean(images_squeeze, axis=-2, keepdims=True)) / (
                        tf.math.reduce_std(images_squeeze, axis=-2, keepdims=True) + 1e-6)
                    images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
                else:
                    images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

                labels = tf.boolean_mask(images_patch, bool_masked_pos, axis=1)

                # Forward pass
                outputs = self.model(images, training=True)
                loss = self.loss_func(labels, outputs)

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            if self.lr_scheduler is not None:
                self.lr_scheduler(it)

            # Logging
            if self.log_writer is not None:
                self.log_writer.update({
                    "loss": loss.numpy(),
                    "lr": self.optimizer.lr.numpy()
                }, step=it)

            self.metric_logger.update(loss=loss.numpy(), lr=self.optimizer.lr.numpy())

        # Sync metrics across processes
        self.metric_logger.synchronize_between_processes()
        print("Averaged stats:", self.metric_logger)
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}

    def _adjust_learning_rate_and_weight_decay(self, iteration, lr_schedule_values, wd_schedule_values):
        if lr_schedule_values is not None:
            tf.keras.backend.set_value(self.optimizer.lr, lr_schedule_values[iteration] * self.optimizer.lr_scale)
        if wd_schedule_values is not None:
            # Assuming there's a way to scale weight decay in the optimizer
            pass
