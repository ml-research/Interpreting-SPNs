from keras import backend as K
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow import set_random_seed
from src.help_functions import *
import matplotlib.pyplot as plt
from src.InterpretableNn import InterpretableNn
from src.influence.dataset import DataSet  # for train and test set creation
import numpy as np

# ==== Influence inspection of a neural net (NN) on the toy color data set ====

n_train = 20000
n_test = 10000
n_val = 1000

seed = 112000  # Random seed
model_name = "nn"
output_path = "output"

t = 2
t_features = None  # [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
t_label = None
n = 1000

np.random.seed(seed)
set_random_seed(seed)
res = 5

(train_samples, train_labels), (test_samples, test_labels) = generate_toy_color(n_train, n_test)

# Split train data into train and validation set
val_samples = train_samples[:n_val]
train_samples = train_samples[n_val:]

val_labels = train_labels[:n_val]
train_labels = train_labels[n_val:]

if t_label is not None:
    # Redefine the label of the test sample
    test_labels[t] = t_label
if t_features is not None:
    # Redefine the feature values of the test sample
    test_samples[t] = t_features

# Adjust the shape
train_labels = np.expand_dims(train_labels, 1)
test_labels = np.expand_dims(test_labels, 1)
val_labels = np.expand_dims(val_labels, 1)

train_set = DataSet(train_samples, train_labels)
test_set = DataSet(test_samples, test_labels)
validation_set = DataSet(val_samples, val_labels)
data_sets = base.Datasets(train=train_set, test=test_set, validation=validation_set)

# Setup the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(5 * 5,), name="Input_Layer"))
model.add(tf.keras.layers.Dense(5 * 5, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2 * 2, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name="Output_Layer"))

model.summary()

# Model configuration
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Learn the model
history = model.fit(train_samples,
                    train_labels,
                    epochs=250,
                    batch_size=100,
                    validation_data=(val_samples, val_labels),
                    verbose=1)
results = model.evaluate(test_samples, test_labels)
print("Test loss:", results[0])
print("Test accuracy:", results[1])

# Get training history
history_dict = history.history

# plot history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

graph = K.get_session().graph
graph.as_default()

# Export the model
export_dir = export_model(root_dir=output_path, export_dir="/nns/tf_" + model_name, force_overwrite=True)
print("Successfully exported SPN tensor to \"%s\"." % export_dir)

tf.reset_default_graph()

# Import the models with new placeholders
sample_placeholder = tf.placeholder(dtype=np.float32,
                                    shape=(1, test_samples.shape[1]),
                                    name="Sample_Placeholder")
label_placeholder = tf.placeholder(dtype=np.float32,
                                   shape=(1, test_labels.shape[1]),
                                   name="Label_Placeholder")

with tf.name_scope("Neural_Net"):
    input_map = {"Input_Layer:0": sample_placeholder,
                 "Output_Layer_target:0": label_placeholder}
    restored_nn_graph = import_model(output_path + "/nns/tf_" + model_name, input_map)
    output_node = restored_nn_graph.get_tensor_by_name("Neural_Net/Output_Layer/Sigmoid:0")
    loss = restored_nn_graph.get_tensor_by_name("Neural_Net/loss/Output_Layer_loss/value:0")

# Create a graph log to visualize the TF graph with TensorBoard
plot_tf_graph(output_node,
              {sample_placeholder: [test_samples[t]],
               label_placeholder: [test_labels[t]]},
              log_dir=output_path + "/logs")

# Plot test sample
plot_toy_color(image=test_samples[t],
               plot_title="Regarded Test Sample",
               label=test_labels[t],
               figsize=2)

# Plot train samples
plot_toy_colors(images=train_samples[0:16],
                plot_title="Train Samples",
                labels=train_labels[0:16])

print('\033[1mStart InterpretableNn class initialization...\033[0m')
start_time = time.time()

interpretable_nn = InterpretableNn(output_node=output_node,
                                   sample_placeholder=sample_placeholder,
                                   label_placeholder=label_placeholder,
                                   loss_op=loss,
                                   data_sets=data_sets,
                                   model_name=model_name,
                                   train_dir=output_path + '/training')

duration = time.time() - start_time
print('\033[1mFinished initialization after %.3f sec.\033[0m' % duration)

# Get IF gradients
influence_grads = interpretable_nn.get_grad_of_influence_wrt_input(test_indices=[t],
                                                                   train_indices=range(n),
                                                                   ignore_hessian=True)

# Plot influence gradient heatmaps
plot_heatmaps(intensities=influence_grads[0:16],
              xdim=res, ydim=res,
              plot_title="Feature Influences w/o Hessian",
              labels=train_labels[0:16],
              rescale=False)

# Sort in descending order
indexlist = np.argsort(np.linalg.norm(influence_grads, axis=1))[::-1]
sorted_inf_grads = influence_grads[indexlist]

# Plot influence gradient heatmaps
plot_heatmaps(intensities=sorted_inf_grads[0:16],
              xdim=res, ydim=res,
              plot_title="Strongest Feature Influences (w/o H.)",
              labels=train_labels[indexlist][0:16])

# Get IF gradients
influence_grads = interpretable_nn.get_grad_of_influence_wrt_input(test_indices=[t],
                                                                   train_indices=range(n),
                                                                   ignore_hessian=False,
                                                                   approx_type='lissa',
                                                                   approx_params={"batch_size": 1,
                                                                                  "scale": 10,
                                                                                  "damping": 0.1,
                                                                                  "num_samples": 1,
                                                                                  "recursion_depth": 50})

# Plot influence gradient heatmaps
plot_heatmaps(intensities=influence_grads[0:16],
              xdim=res, ydim=res,
              plot_title="Feature Influences w/ Hessian",
              labels=train_labels[0:16],
              rescale=False)

# Sort in descending order
indexlist = np.argsort(np.linalg.norm(influence_grads, axis=1))[::-1]
sorted_inf_grads = influence_grads[indexlist]

# Plot influence gradient heatmaps
plot_heatmaps(intensities=sorted_inf_grads[0:16],
              xdim=res, ydim=res,
              plot_title="Strongest Feature Influences (w/ H.)",
              labels=train_labels[indexlist][0:16])
