import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
import collections
import grammar


TrainingResult = collections.namedtuple("TrainingResult", [
    "early_stopping_epoch_number",
    "early_stopping_val_loss_min",
    "initial_W_hid_list",
    "initial_b_hid_list",
    "initial_w_out",
    "initial_b_out",
    "train_loss_list",
    "train_acc_list",
    "test_loss_list",
    "test_acc_list"
])


def print_training_result_summary(tr):
    print()
    early_stopping_test_acc = tr.test_acc_list[tr.early_stopping_epoch_number - 1]
    early_stopping_test_loss = tr.test_loss_list[tr.early_stopping_epoch_number - 1]
    early_stopping_train_acc = tr.train_acc_list[tr.early_stopping_epoch_number - 1]
    early_stopping_train_loss = tr.train_loss_list[tr.early_stopping_epoch_number - 1]
    misclassification_rate = 1 - early_stopping_test_acc
    print("*** Final training result ***")
    print("  best iteration number: {}".format(tr.early_stopping_epoch_number))
    print("  training loss: {:.3f}".format(early_stopping_train_loss))
    print("  training accuracy: {:.3f}".format(early_stopping_train_acc))
    print("  test loss: {:.3f}".format(early_stopping_test_loss))
    print("  test accuracy: {:.3f}".format(early_stopping_test_acc))
    print("  test misclassification rate: {:.3f}".format(misclassification_rate))
    print()


def generate_samples():
    X = []
    X_str = []
    C = []
    for i in range(6000):
        sample_str = grammar.make_reber()
        sample_input_vec = grammar.str_to_vec(sample_str)
        sample_target_vec = grammar.str_to_next(sample_str)
        temp = grammar.vec_to_str(sample_input_vec)
        assert(temp == sample_str)
        X += [sample_input_vec]
        X_str += [sample_str]
        C += [sample_target_vec]

    X_train, X_val, X_test = X[:5000], X[5000:5500], X[5500:]
    C_train, C_val, C_test = C[:5000], C[5000:5500], C[5500:]
    assert len(X_train) == 5000
    assert len(C_train) == 5000
    assert len(X_val) == 500
    assert len(C_val) == 500
    assert len(X_test) == 500
    assert len(C_test) == 500
    return X_train, X_val, X_test, C_train, C_val, C_test


def main():
    ####################################################################################################################
    # Set configuration parameters here!                                                                               #
    ####################################################################################################################

    # number of total iterations for training during validation
    # Note: Although early stopping happens before this defined limit such that the number of epochs with the
    #       minimal cross-entropy error are preserved for final training, the training is further continued until
    #       the defined epoch limit is reached for the purpose of plotting.
    #       The best epoch determined via early stopping is then shown in the plot as a vertical line.
    CONFIG_VALIDATION_NUM_OF_TOTAL_TRAIN_EPOCHS = 600

    # here you can specify the learning rates that should be used during validation
    CONFIG_VALIDATION_LEARNING_RATES = [0.0005, 0.0001, 0.001, 0.002, 0.005, 0.009, 0.01]
    ####################################################################################################################

    # Give the dimension of the data and chose the number of hidden layer
    X_train, X_val, X_test, C_train, C_val, C_test = generate_samples()
    print(X_train[0])
    print(grammar.vec_to_str(X_train[0]))
    import sys;sys.exit()

    # a)
    X, mns, sstd = normalize(X)
    X_tst, _, _ = normalize(X_tst, mns, sstd)

    C = create_one_out_of_k_represantation(C1)
    C_tst = create_one_out_of_k_represantation(C1_tst)

    # fixed seed 42 to generate reproduceable results
    if CONFIG_SHUFFLE_TRAINING_DATA:
        rstate = np.random.RandomState(42)
        shuffled_samples_and_labels = list(zip(X, C))
        rstate.shuffle(shuffled_samples_and_labels)
        X_full_train = np.array(list(map(lambda s: s[0], shuffled_samples_and_labels)))
        C_full_train = np.array(list(map(lambda s: s[1], shuffled_samples_and_labels)))
    else:
        X_full_train = X
        C_full_train = C

    assert (len(X_full_train) == len(C_full_train))
    best_training_result = None
    best_training_misclassification_rate = None
    best_learning_rate = 0.0

    # a) train and evaluate deep network
    print()
    print("=" * 80)
    print()
    print('b) train and evaluate deep network')
    n_hidden = (40, 40, 40, 40, 40, 40, 40, 40, 40)
    for activation_function in ["Tanh", "ReLu"]:
        for learning_rate in CONFIG_VALIDATION_LEARNING_RATES:
            print()
            print("-" * 80)
            print("   LearningRate={}, Architecture={}".format(learning_rate, activation_function))
            print("-" * 80)
            plot_title = "{} - Learn. rate: {}".format(activation_function, learning_rate)

            X_train, C_train = X_full_train, C_full_train
            tr = train_and_evaluate(activation_function, X_train, C_train, X_tst, C_tst, learning_rate,
                                    n_hidden, n_iter=CONFIG_VALIDATION_NUM_OF_TOTAL_TRAIN_EPOCHS)
            misclassification_rate = 1 - tr.test_acc_list[tr.early_stopping_epoch_number - 1]
            print_training_result_summary(tr)
            plot_errors_and_accuracies(plot_title, tr.train_loss_list, tr.train_acc_list,
                                       tr.test_loss_list,
                                       tr.test_acc_list,
                                       tr.early_stopping_epoch_number)

            if best_training_misclassification_rate is None or misclassification_rate < best_training_misclassification_rate:
                best_training_result = tr
                best_training_misclassification_rate = misclassification_rate
                best_learning_rate = learning_rate

        if best_training_misclassification_rate is not None:
            print()
            print('-' * 80)
            print("*** Summary of best model (during validation phase) ***")
            print("   LearningRate={}, Architecture={}".format(best_learning_rate, n_hidden))
            print("   Early stopping: number_of_epochs={}".format(best_training_result.early_stopping_epoch_number))
            print_training_result_summary(best_training_result)

    # b) train and evaluate ResNet
    print()
    print("=" * 80)
    print()
    print('b) train and evaluate ResNet')
    n_hidden = (40, 40, 40, 40, 40, 40, 40, 40, 40)
    for learning_rate in CONFIG_VALIDATION_LEARNING_RATES:
        print()
        print("-" * 80)
        print("   LearningRate={}, Architecture=ResNet".format(learning_rate))
        print("-" * 80)
        plot_title = "ResNet - Learn. rate: {}".format(learning_rate)

        X_train, C_train = X_full_train, C_full_train
        tr = train_and_evaluate_resnet(X_train, C_train, X_tst, C_tst, learning_rate,
                                       n_hidden, n_iter=CONFIG_VALIDATION_NUM_OF_TOTAL_TRAIN_EPOCHS)
        misclassification_rate = 1 - tr.test_acc_list[tr.early_stopping_epoch_number - 1]
        print_training_result_summary(tr)
        plot_errors_and_accuracies(plot_title, tr.train_loss_list, tr.train_acc_list,
                                   tr.test_loss_list,
                                   tr.test_acc_list,
                                   tr.early_stopping_epoch_number)

        if best_training_misclassification_rate is None or misclassification_rate < best_training_misclassification_rate:
            best_training_result = tr
            best_training_misclassification_rate = misclassification_rate
            best_learning_rate = learning_rate

    if best_training_misclassification_rate is not None:
        print()
        print('-' * 80)
        print("*** Summary of best model (during validation phase) ***")
        print("   LearningRate={}, Architecture={}".format(best_learning_rate, n_hidden))
        print("   Early stopping: number_of_epochs={}".format(best_training_result.early_stopping_epoch_number))
        print_training_result_summary(best_training_result)


def train_and_evaluate(activation_function, X_train, C_train, X_test, C_test, learning_rate, n_hidden, n_iter, best_training_result=None):
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
    b_out = tf.Variable(np.zeros(n_out), trainable=True)

    # Define the neuron operations
    x_current = x_input = tf.placeholder(shape=(None, n_in), dtype=tf.float64)
    for W_hid, b_hid in zip(W_hid_list, b_hid_list):
        if activation_function == "Tanh":
            x_current = tf.nn.tanh(tf.matmul(x_current, W_hid) + b_hid)
        else:
            x_current = tf.nn.relu(tf.matmul(x_current, W_hid) + b_hid)
    z_out = tf.matmul(x_current, w_out) + b_out
    z = tf.nn.softmax(z_out)

    z_ = tf.placeholder(shape=(None, n_out), dtype=tf.float64)
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(z_ * tf.log(z), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=z_out, labels=z_))

    # The operation to perform gradient descent.
    # Note that train_step is still a **symbolic operation**, it needs to be executed to update the variables.
    #train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # To evaluate the performance in a readable way, we also compute the classification accuracy.
    correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(z_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    # Open a session and initialize the variables.
    init = tf.global_variables_initializer()  # Create an op that will
    sess = tf.Session()
    sess.run(init)  # Set the value of the variables to their initialization value

    # Create some list to monitor how error decreases
    test_loss_list = []
    train_loss_list = []

    test_acc_list = []
    train_acc_list = []

    # Create minibatches to train faster
    k_batch = 20
    X_batch_list = np.array_split(X_train, k_batch)
    labels_batch_list = np.array_split(C_train, k_batch)

    # early stopping parameters
    early_stopping_val_missclassification_min = np.float64(np.inf)
    early_stopping_epoch_number = 0

    # initialize with the same initial weights
    if best_training_result is not None:
        for idx, (W_hid, b_hid) in enumerate(zip(W_hid_list, b_hid_list)):
            sess.run(tf.assign(W_hid, best_training_result.initial_W_hid_list[idx]))
            sess.run(tf.assign(b_hid, best_training_result.initial_b_hid_list[idx])) # not necessary since it's the zero vector
        sess.run(tf.assign(w_out, best_training_result.initial_w_out))
        sess.run(tf.assign(b_out, best_training_result.initial_b_out)) # not necessary since it's the zero vector
        early_stopping_epoch_number = best_training_result.early_stopping_epoch_number

    # remember initial parameters
    initial_W_hid_list, initial_b_hid_list = [], []
    for W_hid, b_hid in zip(W_hid_list, b_hid_list):
        initial_W_hid_list += [sess.run(W_hid)]
        initial_b_hid_list += [sess.run(b_hid)] # not necessary since it's the zero vector
    initial_w_out = sess.run(w_out)
    initial_b_out = sess.run(b_out) # not necessary since it's the zero vector

    for iteration_idx in range(n_iter):
        # Run gradient steps over each minibatch
        for x_minibatch, labels_minibatch in zip(X_batch_list, labels_batch_list):
            sess.run(train_step, feed_dict={x_input: x_minibatch, z_: labels_minibatch})

        # Compute the errors over the whole dataset
        train_loss = sess.run(cross_entropy, feed_dict={x_input: X_train, z_: C_train})
        test_loss = sess.run(cross_entropy, feed_dict={x_input: X_test, z_: C_test})

        # Compute the acc over the whole dataset
        train_acc = sess.run(accuracy, feed_dict={x_input: X_train, z_: C_train})
        test_acc = sess.run(accuracy, feed_dict={x_input: X_test, z_: C_test})
        test_missclassification_rate = 1.0 - test_acc

        # Put it into the lists
        test_loss_list.append(test_loss)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        # early stopping check BEGIN
        if best_training_result is None:
            if test_missclassification_rate <= early_stopping_val_missclassification_min:
                early_stopping_epoch_number = iteration_idx + 1
                early_stopping_val_missclassification_min = test_missclassification_rate
        else:
            if iteration_idx == (best_training_result.early_stopping_epoch_number - 1):
                early_stopping_val_missclassification_min = test_loss

        if np.mod(iteration_idx, 10) == 0:
            print('iteration {} test accuracy: {:.3f}'.format(iteration_idx + 1, test_acc))

    train_result = TrainingResult(
        early_stopping_epoch_number,
        early_stopping_val_missclassification_min,
        initial_W_hid_list,
        initial_b_hid_list,
        initial_w_out,
        initial_b_out,
        train_loss_list,
        train_acc_list,
        test_loss_list,
        test_acc_list)
    return train_result


def train_and_evaluate_resnet(X_train, C_train, X_test, C_test, learning_rate, n_hidden, n_iter, best_training_result=None):
    assert(len(list(n_hidden)) == 9, "This network should have 9 hidden layers!")
    assert(all(map(lambda n: n == 40, list(n_hidden))), "Each hidden layer must have exactly 40 neurons!")
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
    b_out = tf.Variable(np.zeros(n_out), trainable=True)

    # Define the neuron operations
    x_current = x_input = tf.placeholder(shape=(None, n_in), dtype=tf.float64)

    x_prev = None
    i = 0
    i_prev = -1
    n_residual_blocks = 0
    for W_hid, b_hid in zip(W_hid_list, b_hid_list):
        if i == 0 or (i % 2) == 1:
            x_prev = x_current
            i_prev = i - 1
            print("Normal hidden layer #{}".format(i))
            x_current = tf.nn.relu(tf.matmul(x_current, W_hid) + b_hid)
        else:
            print("Residual block #{} -> previous layer: {}, layer before previous layer: {}".format(n_residual_blocks, (i-1), i_prev))
            x_current = tf.nn.relu(x_prev + tf.matmul(x_current, W_hid) + b_hid)
            n_residual_blocks += 1
        i += 1

    assert(n_residual_blocks == 4, "There must only be 4 residual blocks in this network!")
    z_out = tf.matmul(x_current, w_out) + b_out
    z = tf.nn.softmax(z_out)

    z_ = tf.placeholder(shape=(None, n_out), dtype=tf.float64)

    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(z_ * tf.log(z), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=z_out, labels=z_))

    # The operation to perform gradient descent.
    # Note that train_step is still a **symbolic operation**, it needs to be executed to update the variables.
    #train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # To evaluate the performance in a readable way, we also compute the classification accuracy.
    correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(z_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    # Open a session and initialize the variables.
    init = tf.global_variables_initializer()  # Create an op that will
    sess = tf.Session()
    sess.run(init)  # Set the value of the variables to their initialization value

    # Create some list to monitor how error decreases
    test_loss_list = []
    train_loss_list = []

    test_acc_list = []
    train_acc_list = []

    # Create minibatches to train faster
    k_batch = 20
    X_batch_list = np.array_split(X_train, k_batch)
    labels_batch_list = np.array_split(C_train, k_batch)

    # early stopping parameters
    early_stopping_val_missclassification_min = np.float64(np.inf)
    early_stopping_epoch_number = 0

    # initialize with the same initial weights
    if best_training_result is not None:
        for idx, (W_hid, b_hid) in enumerate(zip(W_hid_list, b_hid_list)):
            sess.run(tf.assign(W_hid, best_training_result.initial_W_hid_list[idx]))
            sess.run(tf.assign(b_hid, best_training_result.initial_b_hid_list[idx])) # not necessary since it's the zero vector
        sess.run(tf.assign(w_out, best_training_result.initial_w_out))
        sess.run(tf.assign(b_out, best_training_result.initial_b_out)) # not necessary since it's the zero vector
        early_stopping_epoch_number = best_training_result.early_stopping_epoch_number

    # remember initial parameters
    initial_W_hid_list, initial_b_hid_list = [], []
    for W_hid, b_hid in zip(W_hid_list, b_hid_list):
        initial_W_hid_list += [sess.run(W_hid)]
        initial_b_hid_list += [sess.run(b_hid)] # not necessary since it's the zero vector
    initial_w_out = sess.run(w_out)
    initial_b_out = sess.run(b_out) # not necessary since it's the zero vector

    for iteration_idx in range(n_iter):
        # Run gradient steps over each minibatch
        for x_minibatch, labels_minibatch in zip(X_batch_list, labels_batch_list):
            sess.run(train_step, feed_dict={x_input: x_minibatch, z_: labels_minibatch})

        # Compute the errors over the whole dataset
        train_loss = sess.run(cross_entropy, feed_dict={x_input: X_train, z_: C_train})
        test_loss = sess.run(cross_entropy, feed_dict={x_input: X_test, z_: C_test})

        # Compute the acc over the whole dataset
        train_acc = sess.run(accuracy, feed_dict={x_input: X_train, z_: C_train})
        test_acc = sess.run(accuracy, feed_dict={x_input: X_test, z_: C_test})
        test_missclassification_rate = 1.0 - test_acc

        # Put it into the lists
        test_loss_list.append(test_loss)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        # early stopping check BEGIN
        if best_training_result is None:
            if test_missclassification_rate <= early_stopping_val_missclassification_min:
                early_stopping_epoch_number = iteration_idx + 1
                early_stopping_val_missclassification_min = test_missclassification_rate
        else:
            if iteration_idx == (best_training_result.early_stopping_epoch_number - 1):
                early_stopping_val_missclassification_min = test_loss

        if np.mod(iteration_idx, 10) == 0:
            print('iteration {} test accuracy: {:.3f}'.format(iteration_idx + 1, test_acc))

    train_result = TrainingResult(
        early_stopping_epoch_number,
        early_stopping_val_missclassification_min,
        initial_W_hid_list,
        initial_b_hid_list,
        initial_w_out,
        initial_b_out,
        train_loss_list,
        train_acc_list,
        test_loss_list,
        test_acc_list)
    return train_result


def plot_errors_and_accuracies(title, train_loss_list, train_acc_list, test_loss_list=None,
                               test_acc_list=None, early_stopping_epoch_number=None):
    fig, ax_list = plt.subplots(1, 2)

    # loss plot
    ax_list[0].plot(train_loss_list, color='blue', label='training', lw=2)

    if test_loss_list is not None:
        ax_list[0].plot(test_loss_list, color='red', label='testing', lw=2)

    # misclassification rate plot
    train_mcr_list = np.subtract(np.ones(len(train_acc_list), dtype=np.float64), train_acc_list)
    ax_list[1].plot(train_mcr_list, color='blue', label='training', lw=2)

    if test_acc_list is not None:
        test_mcr_list = np.subtract(np.ones(len(test_acc_list), dtype=np.float64), test_acc_list)
        ax_list[1].plot(test_mcr_list, color='red', label='testing', lw=2)

    """
    # accuracy plot
    ax_list[1].plot(train_acc_list, color='blue', label='training', lw=2)

    if val_acc_list is not None:
        ax_list[1].plot(val_acc_list, color='green', label='validation', lw=2)

    if test_acc_list is not None:
        ax_list[1].plot(test_acc_list, color='red', label='testing', lw=2)
    """

    ax_list[0].set_xlabel('training iterations')
    ax_list[1].set_xlabel('training iterations')
    ax_list[0].set_ylabel('Cross-entropy')
    ax_list[1].set_ylabel('Misclassification rate')
    #ax_list[1].set_ylabel('Accuracy')

    # shows vertical line where early stopping occurred...
    if early_stopping_epoch_number is not None:
        ax_list[0].axvline(x=(early_stopping_epoch_number - 1))
        ax_list[1].axvline(x=(early_stopping_epoch_number - 1))

    fig.suptitle(title)
    plt.legend(loc=2)
    #plt.subplots_adjust(top=0.85)
    plt.show()


def create_one_out_of_k_represantation(C1):
    C = np.zeros((C1.shape[0], 26))
    for i in range(C1.shape[0]):
        reached = False
        for k in range(0, 26):
            if k == (C1[i] - 1):
                C[i][k] = 1
                reached = True
        assert reached is True, "WHY!?!? {}".format(C1[i])
    for one_hot_label_vector in C:
        assert(np.sum(one_hot_label_vector) == 1.0, "Invalid label vector!")
    return C


if __name__ == "__main__":
    main()

