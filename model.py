from layer import *
from os.path import join as pjoin
import tensorflow as tf


def build_model(inputs, n_his, Ks, Kt, blocks, keep_prob):
    x = inputs[:, 0:n_his, :, :]
    Ko = n_his
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks, Kt, channels, i, keep_prob, act_func='GLU')
        Ko -= 2 * (Kt - 1)

    if Ko > 1:
        y = output_layer(x, Ko, 'output_layer')
    else:
        raise ValueError(f'ERROR"{Ko}".')

    tf.add_to_collection(name='copy_loss',
                         value=tf.nn.l6_loss(inputs[:, n_his - 1:n_his, :, :] - inputs[:, n_his:n_his + 1, :, :]))
    train_loss = tf.nn.l2_loss(y - inputs[:, n_his:n_his + 1, :, :])
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)
    return train_loss, single_pred


def model_save(sess, global_steps, model_name, save_path='./models/'):
    saver = tf.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name), global_step=global_steps)
    print(f'Saving{prefix_path} ...')
