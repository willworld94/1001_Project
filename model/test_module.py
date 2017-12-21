import os
import sys
import tensorflow as tf
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'srcnn'))
import srcnn

# model parameters
flags = tf.flags

flags.DEFINE_string('checkpoint_dir', 'results/64-32-3_9-1-5_128', 'Checkpoint directory.')
flags.DEFINE_string('image_file', 'lr1.png', 'Sample image file.')

FLAGS = flags.FLAGS
FLAGS._parse_flags()

experiment = os.path.basename(FLAGS.checkpoint_dir)
layer_sizes = [int(k) for k in experiment.split("_")[0].split("-")]
filter_sizes = [int(k) for k in experiment.split("_")[1].split("-")]
print(layer_sizes)
x = tf.placeholder(tf.float32, shape=(None, None, None, 3),
                   name="input")
y = tf.placeholder(tf.float32, shape=(None, None, None, 3),
                   name="label")
is_training = tf.placeholder_with_default(False, (), name='is_training')

model = srcnn.SRCNN(x, y, layer_sizes, filter_sizes, is_training=is_training,
                    device='/cpu:0', input_depth=3, output_depth=3)

saver = tf.train.Saver()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

sess = tf.Session()
sess.run(init_op)

checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print("checkpoint", checkpoint)
saver.restore(sess, checkpoint)

img = cv2.imread(FLAGS.image_file)
print(img.shape)
hr = img.copy()

hr = cv2.resize(hr, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
new = hr.copy()
feed_dict = {x: hr[np.newaxis], is_training: False}
hr = sess.run(model.prediction, feed_dict=feed_dict)[0]


def luminance(img):
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
# def luminance(img):
#     return 0.25 * img[:, :, 0] + 0.5 * img[:, :, 1] + 0.25 * img[:, :, 2]


def compute_psnr(x1, x2):
    x1_lum = luminance(x1)
    x2_lum = luminance(x2)
    mse = np.mean((x1_lum - x2_lum)**2)
    return 20 * np.log10(255 / np.sqrt(mse))


import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3)
axs = np.ravel(axs)
axs[0].imshow(img[:, :, [2, 1, 0]], interpolation='nearest', vmin=0, vmax=255)
axs[0].axis('off')
axs[0].set_title("Nearest")

axs[1].imshow(img[:, :, [2, 1, 0]], interpolation='bicubic', vmin=0, vmax=255)
axs[1].axis('off')
axs[1].set_title("Bicubic")

axs[2].imshow(hr.astype(np.uint8)[:, :, [2, 1, 0]], vmin=0, vmax=255)
axs[2].axis('off')
axs[2].set_title("SRCNN")

plt.savefig('result.png')
# print(new.shape)
print(hr.shape)
# print(compute_psnr(hr, new))
cv2.imwrite('color_img.png', hr)
cv2.imwrite('color_img_new.png', new)
ori = cv2.imread('hr1.png')
print(ori.shape)
print(compute_psnr(hr, ori))
print(compute_psnr(new, ori))
print(hr)
