import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam, Adamax
from tensorflow_addons.optimizers import AdamW, Lookahead, NovoGrad, RAdam

def get_optimizer(args, model):
    opt_lower = args.opt.lower()
    learning_rate = args.lr
    weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else 0.

    if 'adam' in opt_lower:
        if 'adamw' in opt_lower:
            optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = Adam(learning_rate=learning_rate)
    elif 'nadam' in opt_lower:
        optimizer = Nadam(learning_rate=learning_rate)
    elif 'sgd' in opt_lower:
        optimizer = SGD(learning_rate=learning_rate, momentum=args.momentum, nesterov='nesterov' in opt_lower)
    elif 'radam' in opt_lower:
        optimizer = RAdam(learning_rate=learning_rate)
    elif 'novograd' in opt_lower:
        optimizer = NovoGrad(learning_rate=learning_rate, weight_decay=weight_decay)
    elif 'rmsprop' in opt_lower:
        optimizer = RMSprop(learning_rate=learning_rate, momentum=args.momentum)
    elif 'lookahead' in opt_lower:
        base_optimizer = SGD(learning_rate=learning_rate, momentum=args.momentum, nesterov='nesterov' in opt_lower)
        optimizer = Lookahead(base_optimizer, sync_period=6, slow_step_size=0.5)
    else:
        raise ValueError("Unsupported optimizer type!")

    return optimizer

# Example usage:
class Args:
    opt = 'adamw'
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.01

args = Args()
model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
optimizer = get_optimizer(args, model)
print(optimizer)
