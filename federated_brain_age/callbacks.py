import math

import tensorflow as tf

# class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
#     # Log the learning rate after each epoch
#     def on_epoch_end(self, epoch, logs=None):
#         lr = self.model.optimizer.lr
#         tf.summary.scalar('learning rate', data=lr, step=epoch)

class DecayingLRSchedule(tf.optimizers.schedules.LearningRateSchedule):
    # Tima based learning rate decay
    def __init__(self, initial_learning_rate, decay, starting_step, decay_step_size):
        self.initial_learning_rate = initial_learning_rate
        self.decay = decay
        self.starting_step = starting_step
        self.decay_step_size = decay_step_size

    def __call__(self, step):
        return self.initial_learning_rate / (1 + self.decay * tf.floor((step + self.starting_step)/self.decay_step_size))
