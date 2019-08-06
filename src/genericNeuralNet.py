from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from scipy.optimize import fmin_ncg

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import backend as K

from src.influence.hessians import hessian_vector_product


class GenericNeuralNet(object):
    """
    Multi-class classification.
    """

    def __init__(self, **kwargs):
        np.random.seed(0)
        tf.set_random_seed(0)

        self.batch_size = kwargs.pop('batch_size')
        self.data_sets = kwargs.pop('data_sets')
        self.train_dir = kwargs.pop('train_dir', 'output')
        self.model_name = kwargs.pop('model_name')

        if 'keep_probs' in kwargs:
            self.keep_probs = kwargs.pop('keep_probs')
        else:
            self.keep_probs = None

        if 'mini_batch' in kwargs:
            self.mini_batch = kwargs.pop('mini_batch')
        else:
            self.mini_batch = True

        if 'damping' in kwargs:
            self.damping = kwargs.pop('damping')
        else:
            self.damping = 0.0

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # Initialize session
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)

        # Setup loss and params
        self.total_loss = self.loss()
        self.params = self.get_all_params()

        # Setup input
        self.sample_placeholder, self.labels_placeholder = self.placeholder_inputs()
        self.v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape(), name="Param_Placeholder_%d" % i) for i, a
                              in enumerate(self.params)]
        self.num_train_examples = self.data_sets.train.labels.shape[0]

        # Setup gradients
        self.grad_total_loss_op = tf.gradients(self.total_loss, self.params,
                                               name="Loss_Gradient_wrt_Params")
        self.grad_loss_wrt_input_op = tf.gradients(self.total_loss, self.sample_placeholder,
                                                   name="Loss_Gradient_wrt_Input")

        # Because tf.gradients auto accumulates, we probably don't need the add_n (or even reduce_sum)
        self.influence_op = tf.add_n(
            [tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b))) for a, b in
             zip(self.grad_total_loss_op, self.v_placeholder)], name="Influence")

        self.grad_influence_wrt_input_op = tf.gradients(self.influence_op, self.sample_placeholder,
                                                        name="Influence_Gradient_wrt_Input")

        # Setup Hessian
        self.hessian_vector = hessian_vector_product(self.total_loss, self.params, self.v_placeholder)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.vec_to_list = self.get_vec_to_list_fn()

    def loss(self):
        pass

    def get_all_params(self):
        pass

    def placeholder_inputs(self):
        pass

    def get_vec_to_list_fn(self):
        params_val = self.sess.run(self.params)

        def vec_to_list(v):
            return_list = []
            cur_pos = 0
            for p in params_val:
                return_list.append(v[cur_pos: cur_pos + len(p)])
                cur_pos += len(p)

            assert cur_pos == len(v), "cur_pos and len(v) must be equal, but are %.f and %.f" % (cur_pos, len(v))
            return return_list

        return vec_to_list

    def reset_datasets(self):
        for data_set in self.data_sets:
            if data_set is not None:
                data_set.reset_batch()

    def fill_feed_dict_with_all_ex(self, data_set):
        feed_dict = {
            self.sample_placeholder: data_set.x,
            self.labels_placeholder: data_set.labels
        }
        return feed_dict

    def fill_feed_dict_with_all_but_one_ex(self, data_set, idx_to_remove):
        num_examples = data_set.x.shape[0]
        idx = np.array([True] * num_examples, dtype=bool)
        idx[idx_to_remove] = False
        feed_dict = {
            self.sample_placeholder: data_set.x[idx, :],
            self.labels_placeholder: data_set.labels[idx]
        }
        return feed_dict

    def fill_feed_dict_with_batch(self, data_set, batch_size=0):
        if batch_size is None:
            return self.fill_feed_dict_with_all_ex(data_set)
        elif batch_size == 0:
            batch_size = self.batch_size

        input_feed, labels_feed = data_set.next_batch(batch_size)
        feed_dict = {
            self.sample_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def fill_feed_dict_with_some_ex(self, data_set, target_indices):
        input_feed = data_set.x[target_indices, :].reshape(len(target_indices), -1)
        labels_feed = data_set.labels[target_indices].reshape(-1)
        print("Input feed:", input_feed)
        print("Labels feed:", labels_feed)
        feed_dict = {
            self.sample_placeholder: input_feed,
            # Edit: self.labels_placeholder: labels_feed,
            self.labels_placeholder: [labels_feed],
        }
        return feed_dict

    def fill_feed_dict_with_one_ex(self, data_set, target_idx):
        input_feed = data_set.x[target_idx, :].reshape(1, -1)
        labels_feed = data_set.labels[target_idx].reshape(-1)
        feed_dict = {
            self.sample_placeholder: input_feed,
            # Edit: self.labels_placeholder: labels_feed,
            self.labels_placeholder: [labels_feed],
        }
        return feed_dict

    def fill_feed_dict_manual(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        input_feed = X.reshape(len(Y), -1)
        labels_feed = Y.reshape(-1)
        feed_dict = {
            self.sample_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def minibatch_mean_eval(self, ops, data_set):

        num_examples = data_set.num_examples
        assert num_examples % self.batch_size == 0
        num_iter = int(num_examples / self.batch_size)

        self.reset_datasets()

        ret = []
        for i in xrange(num_iter):
            feed_dict = self.fill_feed_dict_with_batch(data_set)
            ret_temp = self.sess.run(ops, feed_dict=feed_dict)

            if len(ret) == 0:
                for b in ret_temp:
                    if isinstance(b, list):
                        ret.append([c / float(num_iter) for c in b])
                    else:
                        ret.append([b / float(num_iter)])
            else:
                for counter, b in enumerate(ret_temp):
                    if isinstance(b, list):
                        ret[counter] = [a + (c / float(num_iter)) for (a, c) in zip(ret[counter], b)]
                    else:
                        ret[counter] += (b / float(num_iter))

        return ret

    def update_feed_dict_with_v_placeholder(self, feed_dict, vec):
        for pl_block, vec_block in zip(self.v_placeholder, vec):
            feed_dict[pl_block] = vec_block
        return feed_dict

    def get_inverse_hvp(self, v, approx_type='cg', approx_params=None, verbose=True, output_file=None):
        assert approx_type in ['cg', 'lissa']
        if approx_type == 'lissa':
            return self.get_inverse_hvp_lissa(v, **approx_params, output_file=output_file)
        elif approx_type == 'cg':
            return self.get_inverse_hvp_cg(v, verbose)

    def get_inverse_hvp_lissa(self, v,
                              batch_size=None,
                              scale=10, damping=0.0, num_samples=1, recursion_depth=10000,
                              output_file=None):
        """
        This uses mini-batching; uncomment code for the single sample case.
        """
        inverse_hvp = None
        print_iter = recursion_depth / 10

        for i in range(num_samples):
            # samples = np.random.choice(self.num_train_examples, size=recursion_depth)

            cur_estimate = v

            for j in range(recursion_depth):

                # feed_dict = fill_feed_dict_with_one_ex(
                #   data_set, 
                #   images_placeholder, 
                #   labels_placeholder, 
                #   samples[j])   
                feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, batch_size=batch_size)

                feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, cur_estimate)
                hessian_vector_val = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
                cur_estimate = [a + (1 - damping) * b - c / scale for (a, b, c) in
                                zip(v, cur_estimate, hessian_vector_val)]

                # Update: v + (I - Hessian_at_x) * cur_estimate
                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    norm = np.linalg.norm(np.concatenate([np.asarray(est).flatten() for est in cur_estimate]))
                    output_text = "Recursion at depth %s: norm is %.8lf" % (j, norm)
                    print(output_text)
                    if output_file is not None:
                        output_file.write(output_text + "\n")
                    feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, cur_estimate)

            if inverse_hvp is None:
                inverse_hvp = [b / scale for b in cur_estimate]
            else:
                inverse_hvp = [a + b / scale for (a, b) in zip(inverse_hvp, cur_estimate)]

        inverse_hvp = [a / num_samples for a in inverse_hvp]
        return inverse_hvp

    def minibatch_hessian_vector_val(self, v):

        num_examples = self.num_train_examples
        if self.mini_batch == True:
            batch_size = 100
            assert num_examples % batch_size == 0
        else:
            # Edit: batch_size = self.num_train_examples
            batch_size = self.batch_size

        num_iter = int(num_examples / batch_size)

        self.reset_datasets()
        hessian_vector_val = None
        for i in xrange(num_iter):
            feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, batch_size=batch_size)
            # Can optimize this
            feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, v)
            hessian_vector_val_temp = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
            if hessian_vector_val is None:
                hessian_vector_val = [b / float(num_iter) for b in hessian_vector_val_temp]
            else:
                hessian_vector_val = [a + (b / float(num_iter)) for (a, b) in
                                      zip(hessian_vector_val, hessian_vector_val_temp)]

        hessian_vector_val = [a + self.damping * b for (a, b) in zip(hessian_vector_val, v)]

        return hessian_vector_val

    def get_fmin_loss_fn(self, v):

        def get_fmin_loss(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)

        return get_fmin_loss

    def get_fmin_grad_fn(self, v):
        def get_fmin_grad(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return np.concatenate(hessian_vector_val) - np.concatenate(v)

        return get_fmin_grad

    def get_fmin_hvp(self, x, p):
        hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(p))

        return np.concatenate(hessian_vector_val)

    def get_cg_callback(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)

        def fmin_loss_split(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x), -np.dot(np.concatenate(v), x)

        def cg_callback(x):
            # x is current params
            v = self.vec_to_list(x)
            idx_to_remove = 5  # Mark asks: Why?!

            single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
            train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
            predicted_loss_diff = np.dot(np.concatenate(v),
                                         np.concatenate(train_grad_loss_val)) / self.num_train_examples

            if verbose:
                print('Function value: %s' % fmin_loss_fn(x))
                quad, lin = fmin_loss_split(x)
                print('Split function value: %s, %s' % (quad, lin))
                print('Predicted loss diff on train_idx %s: %s' % (idx_to_remove, predicted_loss_diff))

        return cg_callback

    def get_inverse_hvp_cg(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)
        fmin_grad_fn = self.get_fmin_grad_fn(v)
        cg_callback = self.get_cg_callback(v, verbose)

        fmin_results = fmin_ncg(
            f=fmin_loss_fn,
            x0=np.concatenate(v),
            fprime=fmin_grad_fn,
            fhess_p=self.get_fmin_hvp,
            callback=cg_callback,
            avextol=1e-8,
            maxiter=100)

        return self.vec_to_list(fmin_results)

    def get_test_grad_loss_no_reg_val(self, test_indices, batch_size=200, loss_type='normal_loss'):

        if loss_type == 'normal_loss':
            op = self.grad_total_loss_op
        elif loss_type == 'adversarial_loss':
            op = self.grad_adversarial_loss_op
        else:
            raise ValueError('Loss must be specified')

        if test_indices is not None:
            num_iter = int(np.ceil(len(test_indices) / batch_size))

            test_grad_loss_no_reg_val = None
            for i in range(num_iter):
                start = i * batch_size
                end = int(min((i + 1) * batch_size, len(test_indices)))

                test_feed_dict = self.fill_feed_dict_with_some_ex(self.data_sets.test, test_indices[start:end])

                temp = self.sess.run(op, feed_dict=test_feed_dict)

                if test_grad_loss_no_reg_val is None:
                    test_grad_loss_no_reg_val = [a * (end - start) for a in temp]
                else:
                    test_grad_loss_no_reg_val = [a + b * (end - start) for (a, b) in
                                                 zip(test_grad_loss_no_reg_val, temp)]

            test_grad_loss_no_reg_val = [a / len(test_indices) for a in test_grad_loss_no_reg_val]

        else:
            test_grad_loss_no_reg_val = self.minibatch_mean_eval([op], self.data_sets.test)[0]

        return test_grad_loss_no_reg_val

    # Influence function 2
    def get_influence_on_test_loss(self, test_indices, train_idx,
                                   approx_type='cg', approx_params=None, force_refresh=True, test_description=None,
                                   loss_type='normal_loss',
                                   X=None, Y=None):
        # If train_idx is None then use X and Y (phantom points)
        # Need to make sure test_idx stays consistent between models
        # because mini-batching permutes dataset order

        if train_idx is None:
            if (X is None) or (Y is None): raise ValueError('X and Y must be specified if using phantom points.')
            if X.shape[0] != len(Y): raise ValueError('X and Y must have the same length.')
        else:
            if (X is not None) or (Y is not None): raise ValueError(
                'X and Y cannot be specified if train_idx is specified.')

        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)

        print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (
            self.model_name, approx_type, loss_type, test_description))
        if os.path.exists(approx_filename) and force_refresh == False:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            print('Loaded inverse HVP from %s' % approx_filename)
        else:
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val,
                approx_type,
                approx_params)
            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            print('Saved inverse HVP to %s' % approx_filename)

        duration = time.time() - start_time
        print('Inverse HVP took %s sec' % duration)

        start_time = time.time()
        if train_idx is None:
            num_to_remove = len(Y)
            predicted_loss_diffs = np.zeros([num_to_remove])
            for counter in np.arange(num_to_remove):
                single_train_feed_dict = self.fill_feed_dict_manual(X[counter, :], [Y[counter]])
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp),
                                                       np.concatenate(train_grad_loss_val)) / self.num_train_examples

        else:
            num_to_remove = len(train_idx)
            predicted_loss_diffs = np.zeros([num_to_remove])
            for counter, idx_to_remove in enumerate(train_idx):
                single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp),
                                                       np.concatenate(train_grad_loss_val)) / self.num_train_examples

        duration = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

        return predicted_loss_diffs

    # Influence function 3
    def get_grad_of_influence_wrt_input(self, train_indices, test_indices,
                                        approx_type='cg', approx_params=None, force_refresh=True, verbose=True,
                                        test_description=None,
                                        loss_type='normal_loss'):
        """
        If the loss goes up when you remove a point, then it was a helpful point.
        So positive influence = helpful.
        If we move in the direction of the gradient, we make the influence even more positive, 
        so even more helpful.
        Thus if we want to make the test point more wrong, we have to move in the opposite direction.
        """

        # Calculate v_placeholder (gradient of loss at test point)
        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)

        if verbose: print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (
            self.model_name, approx_type, loss_type, test_description))

        if os.path.exists(approx_filename) and force_refresh == False:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            if verbose: print('Loaded inverse HVP from %s' % approx_filename)
        else:
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val,
                approx_type,
                approx_params,
                verbose=verbose)
            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            if verbose: print('Saved inverse HVP to %s' % approx_filename)

        duration = time.time() - start_time
        if verbose: print('Inverse HVP took %s sec' % duration)

        grad_influence_wrt_input_val = None

        for counter, train_idx in enumerate(train_indices):
            # Put in the train example in the feed dict
            grad_influence_feed_dict = self.fill_feed_dict_with_one_ex(
                self.data_sets.train,
                train_idx)

            self.update_feed_dict_with_v_placeholder(grad_influence_feed_dict, inverse_hvp)

            # Run the grad op with the feed dict
            current_grad_influence_wrt_input_val = \
                self.sess.run(self.grad_influence_wrt_input_op, feed_dict=grad_influence_feed_dict)[0][0, :]

            if grad_influence_wrt_input_val is None:
                grad_influence_wrt_input_val = np.zeros([len(train_indices), len(current_grad_influence_wrt_input_val)])

            grad_influence_wrt_input_val[counter, :] = current_grad_influence_wrt_input_val

        return grad_influence_wrt_input_val
