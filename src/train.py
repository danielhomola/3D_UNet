from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import tensorflow as tf

from src.data_utils import Dataset
from src.network import unet_3d


# -----------------------------------------------------------------------------
#
# setup params
#
# -----------------------------------------------------------------------------

log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

BATCH_SIZE = 2
DEPTH = 4
N_BASE_FILTERS = 16
NUM_CLASSES = 3
CLASS_WEIGHTS= np.array([0.00444413, 0.69519929, 0.30035659])

TRAIN_N = 70
TEST_N = 10
EPOCHS = 100

# -----------------------------------------------------------------------------
#
# setup data
#
# -----------------------------------------------------------------------------


train_dataset = Dataset.load_dataset(
    '../data/processed/train_dataset.pckl'
).create_tf_dataset().shuffle(50).repeat().padded_batch(
    batch_size=BATCH_SIZE,
    padded_shapes=([32, 128, 128, 1], [32, 128, 128, 3]))
test_dataset = Dataset.load_dataset(
    '../data/processed/test_dataset.pckl'
).create_tf_dataset().shuffle(50).repeat().padded_batch(
    batch_size=BATCH_SIZE,
    padded_shapes=([32, 128, 128, 1], [32, 128, 128, 3]))

# setup dataset iterator objects, idea from:
# http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
iterator = tf.data.Iterator.from_structure(
    train_dataset.output_types,
    train_dataset.output_shapes
)
training_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)
next_element = iterator.get_next()

# -----------------------------------------------------------------------------
#
# train network
#
# -----------------------------------------------------------------------------

# as per TF batch_norm docs and also https://goo.gl/1UVeYK
train_phase = tf.placeholder(tf.bool, name="is_training")

# create the neural network model
logits = unet_3d(
    next_element[0],
    training=train_phase,
    depth=DEPTH,
    n_base_filters=N_BASE_FILTERS,
    num_classes=NUM_CLASSES
)


# weighted softmax, see https://stackoverflow.com/a/44563055
class_weights = tf.cast(tf.constant(CLASS_WEIGHTS), tf.float32)
class_weights = tf.reduce_sum(
    tf.cast(next_element[0], tf.float32) * class_weights, axis=-1
)

# setup Adam, as per TF batch_norm docs and also https://goo.gl/1UVeYK
loss = tf.losses.softmax_cross_entropy(
    logits=logits,
    onehot_labels=next_element[1],
    weights=class_weights
)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer().minimize(loss)

# get mean IOU
prediction = tf.argmax(logits, axis=-1)
labels = tf.argmax(next_element[1], -1)
iou, conf_mat = tf.metrics.mean_iou(
    labels=labels,
    predictions=tf.cast(prediction, tf.int32),
    num_classes=NUM_CLASSES
)

# initialise session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
sess.run(training_init_op)
saver = tf.train.Saver()

# run the training
for i in range(EPOCHS * TRAIN_N):
    # get mean IOU, idea taken from: https://stackoverflow.com/a/46414395
    l, _, _, _ = sess.run(
        [loss, train_op, iou, conf_mat],
        feed_dict={train_phase: True}
    )
    miou = sess.run([iou])

    if i % 50 == 0:
        print("Epoch: {}, loss: {:.3f}, training IOU: {:.2f}%".format(
            i, l, miou[0] * 100)
        )

# re-initialize the iterator, but this time with test data
sess.run(test_init_op)
avg_miou = 0
for i in range(TEST_N):
    miou = sess.run([iou], feed_dict={train_phase: False})
    avg_miou += miou

print(
    "Average test set IOU over {} iterations is {:.2f}%".format(
        TEST_N, (avg_miou / TEST_N) * 100))

save_path = saver.save(sess, "/tmp/model.ckpt")
print("Model saved in path: %s" % save_path)
