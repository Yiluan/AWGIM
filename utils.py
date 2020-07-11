import random
import numpy as np
import tensorflow as tf


def save(args):
    if args.data_set == 'mini':
        save_path = 'saved_models_mini/'
    else:
        save_path = 'saved_models_tiered/'
    save_path += 'AWGIM' + str(args.num_way) + '_way_' + str(args.num_shot) + 'shot'
    save_path += '_wd' + str(args.weight_decay) + '_dl' + str(args.dim_latent) + '_lr' + str(args.learning_rate)
    save_path += '_as' + str(args.alpha_1)
    save_path += '_ar' + str(args.alpha_1) + str(args.alpha_2)
    save_path += '/'
    return save_path


def load_batch(args, loader, b, training=True, loader_b=None):
    support_input_total, s_labels_total, query_input_total, q_labels_total = [], [], [], []
    for i in range(b):
        if training:
            if args.more_data:
                a = random.uniform(0., (5. if args.data_set=='mini' else 4.62))
                if a <= (4. if args.data_set=='mini' else 3.62):
                    support_input, s_labels, query_input, q_labels = loader.next_data(args.num_way, args.num_shot, args.num_query)
                else:
                    support_input, s_labels, query_input, q_labels = loader_b.next_data(args.num_way, args.num_shot, args.num_query)
            else:
                support_input, s_labels, query_input, q_labels = loader.next_data(args.num_way, args.num_shot, args.num_query)
        else:
            support_input, s_labels, query_input, q_labels = loader.next_data(args.num_way, args.num_shot, args.num_query)
        support_input_total.append(support_input)
        s_labels_total.append(s_labels)
        query_input_total.append(query_input)
        q_labels_total.append(q_labels)
    support_input = np.concatenate(support_input_total, axis=0)
    s_labels = np.concatenate(s_labels_total, axis=0)
    query_input = np.concatenate(query_input_total, axis=0)
    q_labels = np.concatenate(q_labels_total, axis=0)
    return support_input, s_labels, query_input, q_labels


def clip_gradients(gradients, gradient_threshold, gradient_norm_threshold):
    if gradient_threshold > 0:
        gradients = [
            tf.clip_by_value(g, -gradient_threshold, gradient_threshold)
            for g in gradients
        ]
    if gradient_norm_threshold > 0:
        gradients = [
            tf.clip_by_norm(g, gradient_norm_threshold) for g in gradients
        ]
    return gradients


def grads_and_vars(metatrain_loss, weights, reg_term):
    """Computes gradients of metatrain_loss, avoiding NaN.

    Uses a fixed penalty of 1e-4 to enforce only the l2 regularization (and not
    minimize the loss) when metatrain_loss or any of its gradients with respect
    to trainable_vars are NaN. In practice, this approach pulls the variables
    back into a feasible region of the space when the loss or its gradients are
    not defined.

    Args:
      metatrain_loss: A tensor with the LEO meta-training loss.

    Returns:
      A tuple with:
        metatrain_gradients: A list of gradient tensors.
        metatrain_variables: A list of variables for this LEO model.
    """
    metatrain_variables = weights
    metatrain_gradients = tf.gradients(metatrain_loss, metatrain_variables)

    nan_loss_or_grad = tf.logical_or(
        tf.is_nan(metatrain_loss),
        tf.reduce_any([tf.reduce_any(tf.is_nan(g))
                       for g in metatrain_gradients]))

    regularization_penalty = (1e-4 * reg_term)
    zero_or_regularization_gradients = [
        g if g is not None else tf.zeros_like(v)
        for v, g in zip(tf.gradients(regularization_penalty,
                                     metatrain_variables), metatrain_variables)]

    metatrain_gradients = tf.cond(nan_loss_or_grad,
                                  lambda: zero_or_regularization_gradients,
                                  lambda: metatrain_gradients, strict=True)

    return metatrain_gradients