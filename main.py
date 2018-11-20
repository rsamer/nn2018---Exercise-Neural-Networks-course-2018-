import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
import scipy.stats as stats
from nn18_ex2_load import load_isolet


def main():
    # Import dataset and libraries.
    # Please ignore the deprecation warning while importing the MNIST dataset.


    # Define your variables and the operations that define the tensorflow model.
    # - x,y,z do have have numerical values, those are symbolic **"Tensors"**
    # - x is a matrix and not a vector, is has shape [None,784]. The first dimension correspond to a **batch size**. Multiplying larger matrices is usually faster that multiplying small ones many times, using minibatches allows to process many images in a single matrix multiplication.

    # In[3]:

    # Give the dimension of the data and chose the number of hidden layer
    (X, C1, X_tst, C1_tst) = load_isolet()

    X = stats.zscore(X)
    X_tst = stats.zscore(X)

    C = create_one_out_of_k_represantation(C1)
    C_tst = create_one_out_of_k_represantation(C1_tst)

    n_in = 300
    n_out = 26
    n_hidden = 52

    # Set the variables
    W_hid = tf.Variable(rd.randn(n_in, n_hidden) / np.sqrt(n_in), trainable=True)
    b_hid = tf.Variable(np.zeros(n_hidden), trainable=True)

    w_out = tf.Variable(rd.randn(n_hidden, n_out) / np.sqrt(n_in), trainable=True)
    b_out = tf.Variable(np.zeros(n_out))

    # Define the neuron operations
    x = tf.placeholder(shape=(None, 300), dtype=tf.float64)
    y = tf.nn.tanh(tf.matmul(x, W_hid) + b_hid)
    z = tf.nn.softmax(tf.matmul(y, w_out) + b_out)

    z_ = tf.placeholder(shape=(None, 26), dtype=tf.float64)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(z_ * tf.log(z), reduction_indices=[1]))

    # The operation to perform gradient descent.
    # Note that train_step is still a **symbolic operation**, it needs to be executed to update the variables.

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    # To evaluate the performance in a readable way, we also compute the classification accuracy.

    correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(z_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    # Open a session and initialize the variables.

    init = tf.global_variables_initializer()  # Create an op that will
    sess = tf.Session()
    sess.run(init)  # Set the value of the variables to their initialization value

    # Re init variables to start from scratch
    sess.run(init)

    # Create some list to monitor how error decreases
    test_loss_list = []
    train_loss_list = []

    test_acc_list = []
    train_acc_list = []

    # Create minibtaches to train faster
    k_batch = 40
    X_batch_list = np.array_split(X, k_batch)
    labels_batch_list = np.array_split(C, k_batch)

    for k in range(50):
        # Run gradient steps over each minibatch
        for x_minibatch, labels_minibatch in zip(X_batch_list, labels_batch_list):
            sess.run(train_step, feed_dict={x: x_minibatch, z_: labels_minibatch})

        # Compute the errors over the whole dataset
        train_loss = sess.run(cross_entropy, feed_dict={x: X, z_: C})
        test_loss = sess.run(cross_entropy, feed_dict={x: X_tst, z_: C_tst})

        # Compute the acc over the whole dataset
        train_acc = sess.run(accuracy, feed_dict={x: X, z_: C})
        test_acc = sess.run(accuracy, feed_dict={x: X_tst, z_: C_tst})

        # Put it into the lists
        test_loss_list.append(test_loss)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if np.mod(k, 10) == 0:
            print('iteration {} test accuracy: {:.3f}'.format(k + 1, test_acc))

    # In[25]:

    fig, ax_list = plt.subplots(1, 2)
    ax_list[0].plot(train_loss_list, color='blue', label='training', lw=2)
    ax_list[0].plot(test_loss_list, color='green', label='testing', lw=2)
    ax_list[1].plot(train_acc_list, color='blue', label='training', lw=2)
    ax_list[1].plot(test_acc_list, color='green', label='testing', lw=2)

    ax_list[0].set_xlabel('training iterations')
    ax_list[1].set_xlabel('training iterations')
    ax_list[0].set_ylabel('Cross-entropy')
    ax_list[1].set_ylabel('Accuracy')
    plt.legend(loc=2)


def create_one_out_of_k_represantation(C1):
    C = np.zeros((C1.shape[0], 26))
    for i in range(C1.shape[0]):
        for k in range(0, 25):
            if k == C1[i]:
                C[i][k] = 1
    return C


if __name__ == "__main__":
    main()

