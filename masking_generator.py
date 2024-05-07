import tensorflow as tf

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        # Create a mask with num_mask '1's and (num_patches - num_mask) '0's
        mask = tf.concat([
            tf.zeros(self.num_patches - self.num_mask, dtype=tf.float32),
            tf.ones(self.num_mask, dtype=tf.float32)
        ], axis=0)
        # Shuffle the mask
        mask = tf.random.shuffle(mask)
        return mask

# Usage
random_mask_generator = RandomMaskingGenerator(input_size=(14, 14), mask_ratio=0.75)
mask = random_mask_generator()
print("Generated Mask:", mask.numpy())
