import tensorflow as tf
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import dataset_mini
import dataset_tiered
import model
import random
import utils

parser = argparse.ArgumentParser(description="Few Shot classification")
parser.add_argument('-shot', '--num_shot', type=int, default=1)
parser.add_argument('-way', '--num_way', type=int, default=5)
parser.add_argument('-q', '--num_query', type=int, default=15)
parser.add_argument('-stage', '--stage', type=str, default='train')
parser.add_argument('-sd', '--seed', type=int,    default=1000)

parser.add_argument('-gt', '--gradient_threshold', type=float, default=0.1)
parser.add_argument('-gnt', '--gradient_norm_threshold', type=float, default=0.1)
parser.add_argument("-drop", "--dropout", type=float, default=0.3)
parser.add_argument('-step', '--step_size', type=int, default=15000)
parser.add_argument('-e', '--epoch', type=int, default=500)
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-4)
parser.add_argument("-weight", "--weight_decay", type=float, default=1e-6)

parser.add_argument('-a1', '--alpha_1', type=float, default=1.)
parser.add_argument('-a2', '--alpha_2', type=float, default=0.001)
parser.add_argument('-a3', '--alpha_3', type=float, default=0.001)
parser.add_argument('-sh', '--shuffle', type=int, default=0)

parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument("-g", "--gpu", type=int, default=1)
parser.add_argument('-md', '--more_data', type=bool, default=False)
parser.add_argument('-ds', '--data_set', type=str, default='mini')
parser.add_argument('-dl', '--dim_latent', type=int, default=128)
parser.add_argument('-ms', '--mlp_size', type=int, default=2)

args = parser.parse_args()
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
batch_size = args.batch_size


def main():
    is_training = tf.placeholder(tf.bool, name='is_training')
    num_class = args.num_way
    num_shot = args.num_shot
    num_query = args.num_query

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    support_label = tf.placeholder(tf.int32, (None, ), 'support_label')
    query_label = tf.placeholder(tf.int32, (None, ), 'query_label')

    support_x = tf.placeholder(tf.float32, (None, 640), 'support_x')
    query_x = tf.placeholder(tf.float32, (None, 640), 'query_x')

    support_feature = support_x
    query_feature = query_x
    support_feature = tf.reshape(support_feature, (batch_size, num_class, num_shot, 640))
    query_feature = tf.reshape(query_feature, (batch_size, num_class,  num_query, 640))
    support_label_reshape = tf.reshape(support_label, (batch_size, num_class, num_shot))
    query_label_reshape = tf.reshape(query_label, (batch_size, num_class, num_query))

    awgim = model.AWGIM(args, keep_prob, is_training)
    loss_cls, accuracy, tr_loss, tr_accuracy, support_reconstruction, query_reconstruction = \
        awgim.forward(support_feature, support_label_reshape, query_feature, query_label_reshape)
    reg_term = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name])
    loss_meta = loss_cls + args.alpha_1 * tr_loss + args.alpha_2 * support_reconstruction + args.alpha_3 * query_reconstruction
    Batch = tf.Variable(0, trainable=False, dtype=tf.float32, name='global_step')
    learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate, global_step=Batch,
                                               decay_steps=args.step_size, decay_rate=0.2, staircase=True)
    optim = tf.contrib.opt.AdamWOptimizer(learning_rate=learning_rate, weight_decay=args.weight_decay)
    meta_weights = [v for v in tf.trainable_variables()]
    print(meta_weights)

    if args.stage == 'train':
        meta_gradients = utils.grads_and_vars(loss_meta, meta_weights, reg_term)
        meta_gradients = utils.clip_gradients(meta_gradients, args.gradient_threshold, args.gradient_norm_threshold)
        train_op = optim.apply_gradients(zip(meta_gradients, meta_weights), global_step=Batch)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    save_path = utils.save(args)
    print(save_path)
    os.makedirs(save_path, exist_ok=True)
    if args.stage == 'test':
        print(tf.train.latest_checkpoint(save_path))
        saver.restore(sess, tf.train.latest_checkpoint(save_path))
        print('load model')
    if args.data_set == 'mini':
        loader_train = dataset_mini.dataset_mini('train', args)
        loader_val = dataset_mini.dataset_mini('val', args)
        loader_test = dataset_mini.dataset_mini('test', args)
    else:
        loader_train = dataset_tiered.dataset_tiered('train', args)
        loader_val = dataset_tiered.dataset_tiered('val', args)
        loader_test = dataset_tiered.dataset_tiered('test', args)

    if args.stage == 'train':
        print('Load PKL data')
        loader_train.load_data_pkl()
        loader_val.load_data_pkl()
    else:
        loader_test.load_data_pkl()

    val_best_accuracy = 0.
    n_iter = 0
    record_val_acc = []
    if args.stage == 'train':
        for epoch in range(args.epoch):
            training_accuracy, training_loss, acc_cp, acc_real, c_loss, d_loss, g_loss = [], [], [], [], [], [], []
            # training_loss_cls = []
            for epi in range(100):
                support_input, s_labels, query_input, q_labels = utils.load_batch(args, loader_train, args.batch_size, True, loader_val)
                feed_dict = {support_x: support_input, support_label: s_labels,
                             query_x: query_input, query_label: q_labels,
                             is_training: True, keep_prob: 1. - args.dropout}
                outs = sess.run([train_op, loss_meta, accuracy, Batch], feed_dict=feed_dict)
                training_accuracy.append(outs[2])
                training_loss.append(outs[1])
                n_iter += 1
            if (epoch+1) % 3 == 0:
                log = 'epoch: ', epoch+1, 'accuracy: ', np.mean(training_accuracy), 'loss: ', np.mean(training_loss)
                print(log)
            if (epoch+1) % 3 == 0:
                accuracy_val = []
                loss_val = []
                for epi in range(100):
                    support_input, s_labels, query_input, q_labels = utils.load_batch(args, loader_val, args.batch_size, training=False)
                    outs = sess.run([loss_meta, accuracy, Batch], feed_dict={support_x: support_input, support_label: s_labels,
                                                                                     query_x: query_input, query_label: q_labels,
                                                                                     is_training: False, keep_prob: 1.})
                    accuracy_val.append(outs[1])
                    loss_val.append(outs[0])
                mean_acc = np.mean(accuracy_val)
                std_acc = np.std(accuracy_val)
                ci95 = 1.96 * std_acc / np.sqrt(100)
                print(' Val Acc:{:.4f},std:{:.4f},ci95:{:.4f}'.format(mean_acc, std_acc, ci95), 'at epoch: ', epoch+1)
                record_val_acc.append(mean_acc)
                if mean_acc > val_best_accuracy:
                    val_best_accuracy = mean_acc
                    saver.save(sess, save_path=save_path + 'model.ckpt', global_step=Batch)
            if (epoch + 1) % 100 == 0:
                saver.save(sess, save_path=save_path + 'model.ckpt', global_step=Batch)
    elif args.stage == 'test':
        accuracy_test = []
        loss_test = []
        num = 600
        for epi in range(num):
            support_input, s_labels, query_input, q_labels = utils.load_batch(args, loader_test, args.batch_size, False)
            outs = sess.run([loss_meta, accuracy],
                            feed_dict={support_x: support_input, support_label: s_labels,
                                       query_x: query_input, query_label: q_labels,
                                       is_training: False, keep_prob: 1.})
            accuracy_test.append(outs[1])
            loss_test.append(outs[0])
        mean_acc = np.mean(accuracy_test)
        std_acc = np.std(accuracy_test)
        ci95 = 1.96 * std_acc / np.sqrt(num)
        print('Acc:{:.4f},std:{:.4f},ci95:{:.4f}'.format(mean_acc, std_acc, ci95))

    sess.close()


if __name__ == '__main__':
    main()