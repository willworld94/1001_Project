import os

import sys
import time

import tensorflow as tf
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'srcnn'))
import srcnn

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datacenter'))
from base import SuperResData

# model parameters
flags = tf.app.flags

# model hyperparamters
flags.DEFINE_string('hidden', '64,32,3', 'Number of units in hidden layer 1.')
flags.DEFINE_string('kernels', '9,1,5', 'Kernel size of layer 1.')
flags.DEFINE_integer('depth', 3, 'Number of input channels.')
flags.DEFINE_integer('upscale', 3, 'Upscale factor.')

# Model training parameters
flags.DEFINE_integer('num_epochs', 10000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_string('device', '/cpu:0', 'What device should I train on?')

# where to save things
flags.DEFINE_string('save_dir', 'results/', 'Where to save checkpoints.')
flags.DEFINE_string('log_dir', 'logs/', 'Where to save checkpoints.')
flags.DEFINE_integer('stride', 14, 'The stride to make sub imageset and sub labelset')
flags.DEFINE_integer('patch_size', 33, 'the size of the sub image and sub label')


def _maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def train():
    # checkpoint = "/Users/will/Desktop/DS1001-Intro-to-data-science/projectbestmodel.ckpt"
    with tf.Graph().as_default(), tf.device(FLAGS.device):
        # train_images, train_labels = SuperResData(imageset='BSD100', upscale_factor=FLAGS.upscale).tf_patches(batch_size=FLAGS.batch_size)
        image_obj = SuperResData(imageset='BSD100', upscale_factor=FLAGS.upscale)
        train_images, train_labels = image_obj.make_patches(patch_size=FLAGS.patch_size, stride=FLAGS.stride)
        data_length = len(train_labels)
        print(data_length)
        train_images = np.float32(train_images)
        train_labels = np.float32(train_labels)
        train_images_tensor = tf.constant(train_images, name='train_images', dtype=tf.float32)
        train_labels_tensor = tf.constant(train_labels, name='train_labels', dtype=tf.float32)
        # set placeholders, at test time use placeholder

        is_training = tf.placeholder_with_default(True, (), name='is_training')
        x_placeholder = tf.placeholder_with_default(tf.zeros(shape=(1, 10, 10, 3), dtype=tf.float32),
                                                    shape=(None, None, None, 3),
                                                    name="input_placeholder")
        y_placeholder = tf.placeholder_with_default(tf.zeros(shape=(1, 20, 20, 3), dtype=tf.float32),
                                                    shape=(None, None, None, 3),
                                                    name="label_placeholder")
        x = tf.cond(is_training, lambda: train_images_tensor, lambda: x_placeholder)
        y = tf.cond(is_training, lambda: train_labels_tensor, lambda: y_placeholder)

        # x_interp = tf.image.resize_bicubic(x, [h, w])
        x_interp = tf.minimum(tf.nn.relu(x), 255)

        # build graph
        model = srcnn.SRCNN(x_interp, y, FLAGS.HIDDEN_LAYERS, FLAGS.KERNELS,
                            is_training=is_training, input_depth=FLAGS.depth,
                            output_depth=FLAGS.depth, upscale_factor=FLAGS.upscale,
                            learning_rate=1e-4, device=FLAGS.device)

        # initialize graph
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # Create a session for running operations in the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Initialize the variables (the trained variables and the # epoch counter).
        sess.run(init_op)

        # Start input enqueue threads.
        batch_loss = 0
        summary_loss = [100000]
        update_loss = 0
        counter = 0
        update_check = 20
        whole_loss = []
        for epoch in range(FLAGS.num_epochs):
            batch_inx = (data_length) // FLAGS.batch_size
            for idx in range(batch_inx):
                batch_images = train_images[idx * FLAGS.batch_size: (idx + 1) * FLAGS.batch_size]
                batch_labels = train_labels[idx * FLAGS.batch_size: (idx + 1) * FLAGS.batch_size]
                _, train_loss = sess.run([model.opt, model.loss], feed_dict={x: batch_images, y: batch_labels})
                # print("Step: %i, Index: %i, Train Loss: %2.4f" % (epoch, idx, train_loss))
                counter = counter + 1
                batch_loss += train_loss
                update_loss += train_loss
                whole_loss.append(train_loss)
                if idx % update_check == 0:
                    print("Average loss for this update:", round(update_loss / update_check, 3))
                    update_loss = 0

                if counter % 200 == 0:
                    print("Step: %i, Index: %i, Train Loss: %2.4f" % (epoch, idx, train_loss))

            batch_ave_loss = batch_loss / batch_inx
            print("Epoch: ", epoch, "Batch Loss: ", batch_loss, "Batch Average loss", batch_ave_loss)

            summary_loss.append(batch_loss)
            # If the update loss is at a new minimum, save the model
            if batch_loss <= min(summary_loss):
                print('New Record!')
                print("Step: %i, Batch Loss: %2.4f" % (epoch, batch_loss))
                save_path = saver.save(sess, os.path.join(SAVE_DIR, "bestmodel.ckpt"))
            batch_loss = 0


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS._parse_flags()

    if "gpu" in FLAGS.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device[-1]
        FLAGS.device = '/gpu:0'

    FLAGS.HIDDEN_LAYERS = [int(x) for x in FLAGS.hidden.split(",")]
    FLAGS.KERNELS = [int(x) for x in FLAGS.kernels.split(",")]

    file_dir = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(file_dir, FLAGS.save_dir, "%s_%s_%i" % (
        FLAGS.hidden.replace(",", "-"), FLAGS.kernels.replace(",", "-"),
        FLAGS.batch_size))
    FLAGS.log_dir = os.path.join(file_dir, FLAGS.log_dir)
    _maybe_make_dir(FLAGS.log_dir)
    _maybe_make_dir(os.path.dirname(SAVE_DIR))
    _maybe_make_dir(SAVE_DIR)
    train()
