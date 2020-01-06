"""
Created on March 2, 2019

@author: Mark Rothermel
"""

from src.genericNeuralNet import GenericNeuralNet  # base model class for influence computation

import time  # for timing SPN learning duration
import os
import numpy as np
import tensorflow as tf


class InterpretableSpn(GenericNeuralNet):
    """
    Sum-product-network (SPN) for multi-class classification, interpretable with influence functions.
    """

    def __init__(self, root_node, root_node_marg, sample_placeholder, label_placeholder,
                 ignore_weights, ignore_means, ignore_variances, loss, **kwargs):
        self.root_node = root_node  # root node of SPN
        self.root_node_marg = root_node_marg  # root node of marginalized SPN
        self.sample_placeholder = sample_placeholder
        self.label_placeholder = label_placeholder
        self.ignore_weights = ignore_weights
        self.ignore_means = ignore_means
        self.ignore_variances = ignore_variances
        self.loss = loss

        super().__init__(batch_size=1,
                         mini_batch=False,
                         **kwargs)

    def placeholder_inputs(self):
        """Returns the TensorFlow placeholders for sample and label input."""
        sample_ph = self.sample_placeholder
        label_ph = self.label_placeholder
        return sample_ph, label_ph

    def get_all_params(self):
        """Returns all trainable parameters of the model as a list of tf.Tensors."""
        all_params = tf.trainable_variables()

        # Exclude all "ignored" parameters
        selected_params = all_params
        if self.ignore_weights:
            selected_params = [p for p in selected_params if "weights" not in p.name]
        if self.ignore_means:
            selected_params = [p for p in selected_params if "mean" not in p.name]
        if self.ignore_variances:
            selected_params = [p for p in selected_params if "stdev" not in p.name]

        return selected_params

    def loss(self):
        """Generates the TF computation graph for the loss and returns the loss operator."""

        with tf.name_scope("Loss"):
            # Joint likelihood and marginal likelihood:
            joint_log_likelihood = self.root_node  # P(x, y)
            marginal_log_likelihood = self.root_node_marg  # P(x)

            if self.loss == "joint":
                # Loss defined as the negative joint log-likelihood:
                loss = tf.negative(joint_log_likelihood + 0 * marginal_log_likelihood, name="Total_Loss")
            elif self.loss == "conditional":
                # Loss defined as the negative conditional log-likelihood:
                loss = tf.subtract(marginal_log_likelihood, joint_log_likelihood, name="Total_Loss")
            else:
                raise Exception("Unknown loss specification. Expected 'joint' or 'conditional', but got %s."
                                % self.loss)

        return loss

    # Influence function 2 (taken from influence function repository)
    def get_influence_on_test_loss(self, test_indices,
                                   train_idx,
                                   approx_type='cg',
                                   approx_params=None,
                                   force_refresh=True,
                                   test_description=None,
                                   loss_type='normal_loss',
                                   ignore_training_error=False,
                                   ignore_hessian=False,
                                   output_file=None):
        """Influence function I_{up,loss} with option to ignore Hessian"""

        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)

        print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (
            self.model_name, approx_type, loss_type, test_description))
        if not ignore_hessian:
            if os.path.exists(approx_filename) and force_refresh == False:
                inverse_hvp = list(np.load(approx_filename, allow_pickle=True)['inverse_hvp'])
                print('Loaded inverse HVP from %s' % approx_filename)
            else:
                inverse_hvp = self.get_inverse_hvp(
                    test_grad_loss_no_reg_val,
                    approx_type,
                    approx_params,
                    output_file=output_file)
                np.savez(approx_filename, inverse_hvp=inverse_hvp)
                print('Saved inverse HVP to %s' % approx_filename)
        else:
            inverse_hvp = test_grad_loss_no_reg_val

        duration = time.time() - start_time
        print('Inverse HVP took %s sec' % duration)

        start_time = time.time()

        num_to_remove = len(train_idx)
        predicted_loss_diffs = np.zeros([num_to_remove])
        for counter, idx_to_remove in enumerate(train_idx):

            if not ignore_training_error:
                single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
            else:
                train_grad_loss_val = [np.ones(np.sum([np.prod(np.shape(param)) for param in self.params]))]
            predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp),
                                                   np.concatenate(train_grad_loss_val)) / self.num_train_examples

        duration = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

        return predicted_loss_diffs

    # Influence function 3 (taken from influence function repository and modified)
    def get_grad_of_influence_wrt_input(self, train_indices,
                                        test_indices,
                                        approx_type='cg',
                                        approx_params=None,
                                        force_refresh=True,
                                        verbose=True,
                                        test_description=None,
                                        loss_type='normal_loss',
                                        ignore_hessian=False,
                                        output_file=None):
        """Influence function I_{pert,loss} with option to ignore Hessian"""

        # Calculate v_placeholder (gradient of loss at test point)
        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)

        if verbose: print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (
            self.model_name, approx_type, loss_type, test_description))

        if not ignore_hessian:
            if os.path.exists(approx_filename) and force_refresh == False:
                inverse_hvp = list(np.load(approx_filename, allow_pickle=True)['inverse_hvp'])
                if verbose: print('Loaded inverse HVP from %s' % approx_filename)
            else:
                inverse_hvp = self.get_inverse_hvp(
                    test_grad_loss_no_reg_val,
                    approx_type,
                    approx_params,
                    verbose=verbose,
                    output_file=output_file)
                np.savez(approx_filename, inverse_hvp=inverse_hvp)
                if verbose: print('Saved inverse HVP to %s' % approx_filename)
        else:
            inverse_hvp = test_grad_loss_no_reg_val

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
                grad_influence_wrt_input_val = np.zeros(
                    [len(train_indices), len(current_grad_influence_wrt_input_val)])

            grad_influence_wrt_input_val[counter, :] = current_grad_influence_wrt_input_val

        return grad_influence_wrt_input_val

    def get_grad_loss_wrt_input(self, test_indices):
        """Gets a list of test sample indices and returns a list
        of loss gradients regarding the inputs of each of the test samples."""
        grads = []
        op = self.grad_loss_wrt_input_op

        # For each test sample get the loss gradient
        for i in test_indices:
            test_sample = self.data_sets.test.x[i]
            test_label = self.data_sets.test.labels[i]
            feed_dict = {"Sample_Placeholder:0": [test_sample],
                         "Label_Placeholder:0": [test_label]}
            gradient = self.sess.run(op, feed_dict=feed_dict)
            grads = np.append(grads, gradient)

        d = self.grad_loss_wrt_input_op[0].shape[1].value
        return np.reshape(grads, (-1, d))
