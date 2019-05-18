"""
Created on March 2, 2019

@author: Mark Rothermel
"""

from src.influence.genericNeuralNet import GenericNeuralNet  # base model class for influence computation

import time  # for timing SPN learning duration
import os
import numpy as np
import tensorflow as tf


class InterpretableSpn(GenericNeuralNet):
    """
    Sum-product-network (SPN) for multi-class classification, interpretable with influence functions.
    """

    def __init__(self, root_node, input_placeholder, label_placeholder, label_idx, num_epochs, **kwargs):
        self.root_node = root_node
        self.input_placeholder = input_placeholder
        self.label_placeholder = label_placeholder
        self.label_idx = label_idx

        self.inference_needs_labels = True
        self.num_epochs = num_epochs

        super().__init__(initial_learning_rate=0.001,  # ? TODO
                         decay_epochs=[1000, 10000],  # ? TODO
                         **kwargs)

        '''assert self.input_dim == np.prod(self.data_sets.train.x[0].shape), \
            "Input dimension %d does not match with dimension %s of train samples." % \
            (self.input_dim, np.prod(self.data_sets.train.x[0].shape))
        assert self.input_dim == np.prod(self.data_sets.test.x[0].shape), \
            "Input dimension %d does not match with dimension %s of test samples." % \
            (self.input_dim, np.prod(self.data_sets.test.x[0].shape))'''

    def placeholder_inputs(self):
        """Generates TensorFlow placeholders for input and labels."""
        sample_ph = self.input_placeholder
        label_ph = self.label_placeholder
        return sample_ph, label_ph

    def inference(self, sample_ph, label_ph):
        """Gets an input placeholder and returns the root tensor of the SPN."""
        root_tensor = tf.reshape(self.root_node, [self.batch_size, -1])

        return root_tensor

    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds

    def get_all_params(self):
        """Returns all trainable parameters of the model as a list of tf.Tensors."""
        all_params = tf.trainable_variables()
        return all_params

    def loss(self, logits, labels):
        """Computes for a given list of logits the sum of all losses in total,
        the mean of all the losses and the list of individual losses."""
        # The logits already incorporate the cross entropy
        cross_entropy = logits

        # List of individual losses
        indiv_loss_no_reg = cross_entropy

        # Mean of the individual losses, i.e. mean of the given cross entropy values
        loss_no_reg = tf.reduce_mean(cross_entropy, name='loss_mean')
        tf.add_to_collection('losses', loss_no_reg)

        # Total loss, i.e. sum of all individual losses
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return total_loss, loss_no_reg, indiv_loss_no_reg

    def predict_class_of(self, samples, sess):
        """Takes an unlabeled sample and predicts its class.
        Returns the predicted (list of) class(es)."""
        # Setup input
        input_ph = tf.placeholder(tf.float32, samples.shape)
        output_tensor = self.model.forward(input_ph)  # TODO
        # Compute model output
        output = sess.run(output_tensor, feed_dict={input_ph: samples})
        # Interpret output
        max_idx = np.argmax(output, axis=1)
        return max_idx

    def compute_acc(self, samples, labels, sess):
        """Computes the accuracy of the model performing on a given set of samples with a belonging set of labels."""
        # Compute predictions
        predictions = self.predict_class_of(samples, sess)
        # Compute accuracy
        num_correct = np.sum(predictions == labels)
        acc = num_correct / len(samples)
        return acc

    # Influence function 3
    def get_influence_on_test_loss(self, test_indices,
                                   train_idx,
                                   approx_type='cg',
                                   approx_params=None,
                                   force_refresh=True,
                                   test_description=None,
                                   loss_type='normal_loss',
                                   ignore_training_error=False,
                                   ignore_hessian=False):
        """Influence function I_{pert,loss}"""

        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)

        print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (
            self.model_name, approx_type, loss_type, test_description))
        if not ignore_hessian:
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
                train_grad_loss_val = [
                    -(self.data_sets.train.labels[idx_to_remove] * 2 - 1) * self.data_sets.train.x[idx_to_remove, :]]
            predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp),
                                                   np.concatenate(train_grad_loss_val)) / self.num_train_examples

        duration = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

        return predicted_loss_diffs
