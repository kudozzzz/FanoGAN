"""
Anomaly scoring

Copyright (c) 2018 Thomas Schlegl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys
import pickle
import csv
import re
import time

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

from wgangp_64x64 import GoodGenerator, GoodDiscriminator, ResidualBlock
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.img_loader


class bcolors:
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'


ZDIM = 128
DIM = 64  # Model dimensionality
BATCH_SIZE = 64
OUTPUT_DIM = 64 * 64 * 1  # Number of pixels in each image
N_GPUS = 1

print(bcolors.GREEN + "\n=== ANOMALY SCORING PARAMETERS ===" + bcolors.ENDC)
lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]


def l2_norm(x, y, axis=None):
    return tf.reduce_sum(tf.pow(x - y, 2), axis=axis)


def MSE(x, y, axis=None):
    return tf.reduce_mean(tf.pow(x - y, 2), axis=axis)


def load(session, saver, checkpoint_dir, checkpoint_iter=None):
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        if checkpoint_iter is not None:
            last_ckpt_iter = re.match(r'.*.model-(\d+)', ckpt.model_checkpoint_path).group(1)
            target_ckpt_path = re.sub('model-%s' % last_ckpt_iter, 'model-%d' % checkpoint_iter,
                                      ckpt.model_checkpoint_path)
            saver.restore(session, target_ckpt_path)
            ckpt_name = os.path.basename(target_ckpt_path)
        else:
            saver.restore(session, ckpt.model_checkpoint_path)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        return True, ckpt_name
    else:
        return False, ''


def anomaly_scoring(checkpoint_dir, checkpoint_iter, dual_iloss=True):
    rand_sampling = 'normal' if '_norm' in checkpoint_dir else 'unif'
    loss_type = 'MSE' if 'MSE' in checkpoint_dir else ('l2Mean' if 'l2Mean' in checkpoint_dir else 'l2Sum')
    kappa = float(re.match(r'.*MSE-k(\d\.\d{2})', checkpoint_dir).group(1)) if 'MSE' in checkpoint_dir else None

    suff_txt = '_dil' if dual_iloss else ''
    z_reg_type = next((z for z in ['3s_tanh_fc', '05s_tanh_fc', 'tanh_fc', '3s_hard_clip', '05s_hard_clip', 'hard_clip', 'stoch_clip'] if z in checkpoint_dir), None)
    print(bcolors.YELLOW + f"\nUSING z_reg_type='{z_reg_type}'!\n" + bcolors.ENDC)

    print(bcolors.GREEN + f"\nmapping_via_encoder:: checkpoint_dir: {checkpoint_dir}\ncheckpoint_iter: {checkpoint_iter}\n" + bcolors.ENDC)
    model_type_name = checkpoint_dir.replace('z_encoding_d/', '').replace('/checkpoints', '') + suff_txt
    mapping_path = os.path.join('mappings', model_type_name)
    os.makedirs(mapping_path, exist_ok=True)
    log_meta_path = os.path.join(mapping_path, f'mapping_results-enc_ckpt_it{checkpoint_iter}.csv')

    test_gen, ano_gen = lib.img_loader.load(BATCH_SIZE, 'anomaly_score')
    nr_mapping_imgs = lib.img_loader.get_nr_test_samples(BATCH_SIZE)
    Generator = GoodGenerator
    Discriminator = GoodDiscriminator

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': len(DEVICES)}, allow_soft_placement=True)) as session:
        real_data = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1, 64, 64])
        real_data_norm = tf.reshape(2 * ((tf.cast(real_data, tf.float32) / 255.) - .5), [BATCH_SIZE, OUTPUT_DIM])
        emb_query = Encoder(real_data_norm, is_training=False, z_reg_type=z_reg_type, rand_sampling=rand_sampling)
        recon_img = Generator(BATCH_SIZE, noise=emb_query, rand_sampling='normal', is_training=False)
        _, recon_features = Discriminator(recon_img, is_training=False)
        _, image_features = Discriminator(real_data_norm, is_training=False, reuse=True)
        z_img_emb_query = Encoder(recon_img, is_training=False, reuse=True, z_reg_type=z_reg_type, rand_sampling=rand_sampling)

        img_distance = l2_norm(real_data_norm, recon_img, axis=1) if loss_type != 'MSE' else MSE(real_data_norm, recon_img, axis=1)
        z_distance = l2_norm(emb_query, z_img_emb_query, axis=1) if loss_type != 'MSE' else MSE(emb_query, z_img_emb_query, axis=1)
        if loss_type == 'MSE' and dual_iloss:
            loss_fts = MSE(recon_features, image_features, axis=1)
            img_distance += kappa * loss_fts

        saver = tf.train.Saver(max_to_keep=15)
        session.run(tf.global_variables_initializer())
        isLoaded, ckpt = load(session, saver, checkpoint_dir, checkpoint_iter)
        assert isLoaded

        start_time = time.time()
        if os.path.isfile(log_meta_path):
            os.remove(log_meta_path)

        encodings = {'target': [], 'zs': []}
        img_dists, z_dists = [], []
        for is_anom, _gen in enumerate([test_gen(), ano_gen()]):
            for _idx in range(nr_mapping_imgs[is_anom] // BATCH_SIZE):
                (_data,) = next(_gen)
                _img_dist, _dist_z, _z = session.run([img_distance, z_distance, emb_query],
                                                     feed_dict={real_data: _data})
                with open(log_meta_path, "a") as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerows([[is_anom, di, dz] for di, dz in zip(_img_dist, _dist_z)])
                encodings['target'].append(is_anom)
                encodings['zs'].append(_z)
                img_dists.append(_img_dist)
                z_dists.append(_dist_z)
        print(f"\nDONE! (mapping took {time.time() - start_time:.1f} seconds.)\n")
        with open(log_meta_path.replace('.csv', '.pkl'), 'wb') as f:
            pickle.dump({'encodings': encodings, 'img_dists': img_dists, 'z_dists': z_dists}, f, pickle.HIGHEST_PROTOCOL)
        print("Done!\n")


if __name__ == '__main__':
    anomaly_scoring("path-to-'checkpoints'-folder-of-encoder-training", <checkpoint-iter>)
