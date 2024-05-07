import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import L2

def drop_path(inputs, drop_prob, is_training):
    if drop_prob == 0. or not is_training:
        return inputs
    keep_prob = 1 - drop_prob
    shape = (inputs.shape[0],) + (1,) * (len(inputs.shape) - 1)
    random_tensor = keep_prob + tf.random.uniform(shape, dtype=inputs.dtype)
    random_tensor = tf.floor(random_tensor)
    return (inputs / keep_prob) * random_tensor

class MLP(layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation=layers.Activation('gelu'), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features)
        self.act = activation
        self.fc2 = layers.Dense(out_features)
        self.drop = layers.Dropout(drop)

    def call(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x

class Attention(layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, inputs, training=False):
        B, N, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        qkv = self.qkv(inputs)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ tf.transpose(k, [0, 1, 3, 2])) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = (attn @ v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B, N, -1])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x

class TransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path_rate=0., norm_layer=layers.LayerNormalization, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(epsilon=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = partial(drop_path, drop_prob=drop_path_rate)
        self.norm2 = norm_layer(epsilon=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def call(self, x, training=False):
        x = x + self.drop_path(self.attn(self.norm1(x)), training)
        x = x + self.drop_path(self.mlp(self.norm2(x), training=training), training)
        return x

class VisionTransformer(Model):
    def __init__(self, img_size=224, patch_size=16, num_layers=12, num_heads=12, 
                 mlp_ratio=4., channels=3, num_classes=1000, embed_dim=768, 
                 qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=layers.LayerNormalization, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')
        self.pos_embed = self.add_weight('pos_embed', shape=(1, self.num_patches + 1, embed_dim), 
                                         initializer=RandomNormal(0.02), trainable=True)
        self.cls_token = self.add_weight('cls_token', shape=(1, 1, embed_dim), 
                                         initializer=RandomNormal(0.02), trainable=True)
        self.dropout = layers.Dropout(drop_rate)

        self.blocks = [TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, drop_path_rate*i/num_layers, norm_layer)
                       for i in range(num_layers)]
        self.norm = norm_layer(epsilon=1e-6)
        self.head = layers.Dense(num_classes)

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        x = self.patch_embed(x)
        x = tf.reshape(x, [B, -1, self.embed_dim])

        cls_tokens = tf.broadcast_to(self.cls_token, [B, 1, self.embed_dim])
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embed
        x = self.dropout(x, training=training)

        for blk in self.blocks:
            x = blk(x, training=training)

        x = self.norm(x)
        x = self.head(x[:, 0])
        return x
