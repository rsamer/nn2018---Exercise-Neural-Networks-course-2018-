import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from nn18_ex2_load import load_isolet
import itertools
import random


def compute_label_frequencies(C1):
    frequencies = {}
    for i, val in enumerate(C1):
        if val not in frequencies:
            frequencies[val] = 0
        frequencies[val] += 1
    return frequencies


def normalize(X, mns=None, sstd=None):
    '''
    based on scipy.stats.stats.zscore
    '''
    X = np.asanyarray(X)
    mns = X.mean(axis=0) if mns is None else mns
    sstd = X.std(axis=0, ddof=0) if sstd is None else sstd
    normalized_X = (X - mns) / sstd
    return normalized_X, mns, sstd


def main():
    # Import dataset and libraries.
    # Please ignore the deprecation warning while importing the MNIST dataset.


    # Define your variables and the operations that define the tensorflow model.
    # - x,y,z do have have numerical values, those are symbolic **"Tensors"**
    # - x is a matrix and not a vector, is has shape [None,784]. The first dimension correspond to a **batch size**. Multiplying larger matrices is usually faster that multiplying small ones many times, using minibatches allows to process many images in a single matrix multiplication.

    # In[3]:

    # Give the dimension of the data and chose the number of hidden layer
    (X, C1, X_tst, C1_tst) = load_isolet()

    X, mns, sstd = normalize(X)
    X_tst, _, _ = normalize(X_tst, mns, sstd)

    print('-' * 80)
    print(X.shape)
    print(C1.shape)
    print(X_tst.shape)
    print(C1_tst.shape)
    print(compute_label_frequencies(C1))
    print(compute_label_frequencies(C1_tst))

    C = create_one_out_of_k_represantation(C1)
    C_tst = create_one_out_of_k_represantation(C1_tst)
    print(C.shape)
    print(C_tst.shape)
    """
    print(C1[40:60])
    print(C[40:60])
    print('-' * 80)
    import sys;sys.exit()
    """

    # fixed seed 42 to generate reproduceable results
    rstate = np.random.RandomState(42)
    shuffled_samples_and_labels = list(zip(X, C))
    rstate.shuffle(shuffled_samples_and_labels)
    X_full_train = np.array(list(map(lambda s: s[0], shuffled_samples_and_labels)))
    C_full_train = np.array(list(map(lambda s: s[1], shuffled_samples_and_labels)))

    k = 3
    val_acc_list, val_loss_list = [], []

    #for learning_rate in [0.05, 0.1, 0.5]:
    for learning_rate in [0.1]:
        # here you can easily specify how many hidden layers the network should have as well as how many neurons the hidden layers should have
        for n_hidden in [
            (52,),            # 1 hidden layer  (hiddenLayer1: 52 neurons)
            (104, 52),        # 2 hidden layers (hiddenLayer1: 104 neurons, hiddenLayer2: 52 neurons)
            (208, 104, 52),   # 3 hidden layers (hiddenLayer1: 208 neurons, hiddenLayer2: 104 neurons, hiddenLayer3: 52 neurons)
            #(104, 52, 32),    # 3 hidden layers (hiddenLayer1: 104 neurons, hiddenLayer2: 52 neurons, hiddenLayer3: 32 neurons)
            #(208, 52),        # 2 hidden layers (hiddenLayer1: 208 neurons, hiddenLayer2: 52 neurons)
            #(52, 26, 26),     # 3 hidden layers (hiddenLayer1: 52 neurons, hiddenLayer2: 26 neurons, hiddenLayer3: 26 neurons)
            #(104, 104),       # 2 hidden layers (hiddenLayer1: 104 neurons, hiddenLayer2: 104 neurons)
            #(208, 208),       # 2 hidden layers (hiddenLayer1: 208 neurons, hiddenLayer2: 208 neurons)
            #(52, 52, 52),     # 3 hidden layers (hiddenLayer1: 52 neurons, hiddenLayer2: 52 neurons, hiddenLayer3: 52 neurons)
            #(104, 104, 104)   # 3 hidden layers (hiddenLayer1: 104 neurons, hiddenLayer2: 104 neurons, hiddenLayer3: 104 neurons)
        ]:
            for round in range(k):
                ########################################################################################################
                # NOTE: *Stratified* k-fold CV does NOT need to be applied here.                                       #
                #       Normal k-fold CV is sufficient since all labels are already almost equally distributed         #
                ########################################################################################################
                n_train_samples = len(X_full_train)
                n_fold_samples = int(n_train_samples / k)
                n_last_fold_samples = n_train_samples - (n_fold_samples * (k - 1))
                X_train, C_train = None, None

                for i in range(k):
                    n_current_fold_samples = n_fold_samples if i < (k - 1) else n_last_fold_samples
                    start_index = i * n_current_fold_samples
                    end_index = start_index + n_current_fold_samples
                    print("{}-{}".format(start_index, end_index))
                    #---------------------------------------------------------------------------------------------------
                    #
                    #        TODO: @RALPH: review and make sure that the split is really exact and correct!!!!!!!!
                    #
                    #---------------------------------------------------------------------------------------------------
                    if i == round:
                        X_val, C_val = X_full_train[start_index:end_index], C_full_train[start_index:end_index]
                    else:
                        if X_train is not None:
                            X_train = np.concatenate((X_train, X_full_train[start_index:end_index]), axis=0)
                            C_train = np.concatenate((C_train, C_full_train[start_index:end_index]), axis=0)
                        else:
                            X_train = X_full_train[start_index:end_index]
                            C_train = C_full_train[start_index:end_index]

                val_acc, val_loss = train_and_evalate(X_train, C_train, X_val, C_val, learning_rate, n_hidden)
                val_acc_list += [val_acc]
                val_loss_list += [val_loss]
            val_mean = np.mean(val_acc_list)
            val_std = np.std(val_acc_list)
            print("Parameters: LearningRate={}, Architecture={}".format(learning_rate, n_hidden))
            print("Mean: {:.3f}, STD: {:.3f}".format(val_mean, val_std))
            print(val_mean)
            print(val_std)


def train_and_evalate(X_train, C_train, X_val, C_val, learning_rate, n_hidden):
    n_in = 300
    n_out = 26

    # Set the variables
    n_previous = n_in
    W_hid_list, b_hid_list = [], []

    for n_current_hidden in n_hidden:
        W_hid_list += [tf.Variable(rd.randn(n_previous, n_current_hidden) / np.sqrt(n_in), trainable=True)]
        b_hid_list += [tf.Variable(np.zeros(n_current_hidden), trainable=True)]
        n_previous = n_current_hidden

    w_out = tf.Variable(rd.randn(n_previous, n_out) / np.sqrt(n_in), trainable=True)
    b_out = tf.Variable(np.zeros(n_out))

    # Define the neuron operations
    x_current = x_input = tf.placeholder(shape=(None, n_in), dtype=tf.float64)
    for W_hid, b_hid in zip(W_hid_list, b_hid_list):
        x_current = tf.nn.tanh(tf.matmul(x_current, W_hid) + b_hid)
    z = tf.nn.softmax(tf.matmul(x_current, w_out) + b_out)

    z_ = tf.placeholder(shape=(None, n_out), dtype=tf.float64)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(z_ * tf.log(z), reduction_indices=[1]))

    # The operation to perform gradient descent.
    # Note that train_step is still a **symbolic operation**, it needs to be executed to update the variables.
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

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
    #test_loss_list = []
    val_loss_list = []
    train_loss_list = []

    #test_acc_list = []
    val_acc_list = []
    train_acc_list = []

    # Create minibatches to train faster
    k_batch = 40
    X_batch_list = np.array_split(X_train, k_batch)
    #labels_batch_list = np.array_split(C, k_batch)
    labels_batch_list = np.array_split(C_train, k_batch)

    # early stopping parameters
    val_loss_min = np.float64(np.inf)
    num_of_iterations_where_val_loss_increased_after_minimum = 0
    max_num_of_iterations_where_val_loss_increased_after_minimum = 70
    early_stopping_best_iteration_number = 0

    for iteration_idx in itertools.count():
        # Run gradient steps over each minibatch
        for x_minibatch, labels_minibatch in zip(X_batch_list, labels_batch_list):
            sess.run(train_step, feed_dict={x_input: x_minibatch, z_: labels_minibatch})

        # Compute the errors over the whole dataset
        train_loss = sess.run(cross_entropy, feed_dict={x_input: X_train, z_: C_train})
        val_loss = sess.run(cross_entropy, feed_dict={x_input: X_val, z_: C_val})
        #test_loss = sess.run(cross_entropy, feed_dict={x_input: X_tst, z_: C_tst})

        # ----------------------------------------------------------------------------------------------------------
        # early stopping check BEGIN
        # ----------------------------------------------------------------------------------------------------------
        if val_loss <= val_loss_min:
            val_loss_min = val_loss
            num_of_iterations_where_val_loss_increased_after_minimum = 0
        else:
            num_of_iterations_where_val_loss_increased_after_minimum += 1

        if num_of_iterations_where_val_loss_increased_after_minimum > max_num_of_iterations_where_val_loss_increased_after_minimum:
            early_stopping_best_iteration_number = iteration_idx - (num_of_iterations_where_val_loss_increased_after_minimum + 1)
            break
        # ----------------------------------------------------------------------------------------------------------
        # early stopping check END
        # ----------------------------------------------------------------------------------------------------------

        # Compute the acc over the whole dataset
        train_acc = sess.run(accuracy, feed_dict={x_input: X_train, z_: C_train})
        val_acc = sess.run(accuracy, feed_dict={x_input: X_val, z_: C_val})
        #test_acc = sess.run(accuracy, feed_dict={x_input: X_tst, z_: C_tst})

        # Put it into the lists
        #test_loss_list.append(test_loss)
        val_loss_list.append(val_loss)
        train_loss_list.append(train_loss)
        #test_acc_list.append(test_acc)
        val_acc_list.append(val_acc)
        train_acc_list.append(train_acc)

        #print('iteration {} validation accuracy/loss: {:.3f}/{:.3f}'.format(iteration_idx + 1, val_acc, val_loss))
        if np.mod(iteration_idx, 10) == 0:
            print('iteration {} validation accuracy: {:.3f}'.format(iteration_idx + 1, val_acc))
            #print('iteration {} test accuracy: {:.3f}'.format(k + 1, test_acc))

    evaluated_val_acc = val_acc_list[early_stopping_best_iteration_number]
    evaluated_val_loss = val_loss_list[early_stopping_best_iteration_number]
    print('best iteration number according to early stopping is: {}'.format(early_stopping_best_iteration_number+1))
    print('validation accuracy was: {:.3f}'.format(evaluated_val_acc))
    print('validation loss was: {:.3f}'.format(evaluated_val_loss))

    # In[25]:
    # TODO: show vertical line where early stopping occurred...
    fig, ax_list = plt.subplots(1, 2)
    ax_list[0].plot(train_loss_list, color='blue', label='training', lw=2)
    ax_list[0].plot(val_loss_list, color='green', label='validation', lw=2)
    #ax_list[0].plot(test_loss_list, color='green', label='testing', lw=2)
    ax_list[1].plot(train_acc_list, color='blue', label='training', lw=2)
    ax_list[1].plot(val_acc_list, color='green', label='validation', lw=2)
    #ax_list[1].plot(test_acc_list, color='green', label='testing', lw=2)

    ax_list[0].set_xlabel('training iterations')
    ax_list[1].set_xlabel('training iterations')
    ax_list[0].set_ylabel('Cross-entropy')
    ax_list[1].set_ylabel('Accuracy')
    plt.legend(loc=2)
    plt.show()
    return evaluated_val_acc, evaluated_val_loss


def create_one_out_of_k_represantation(C1):
    C = np.zeros((C1.shape[0], 26))
    for i in range(C1.shape[0]):
        reached = False
        for k in range(0, 26):
            if k == (C1[i] - 1):
                #print(k)
                C[i][k] = 1
                reached = True
        if reached is False:
            print('WHY!?!? {}'.format(C1[i]))
            import sys;sys.exit()
    for one_hot_label_vector in C:
        assert(np.sum(one_hot_label_vector) == 1.0, "Invalid label vector!")
    return C


if __name__ == "__main__":
    main()

