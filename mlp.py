# from audioop import cross
import os
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt


# from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

def relu(x, deriv=False):
    # ReLU activation function
    if (deriv):
        return np.where(x <= 0, 0, 1)
    return np.maximum(x, 0)


def softmax(x):
    '''
        softmax(x) = exp(x) / sum(exp(x))
    '''

    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    prob = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return prob


def cross_entropy_loss(predicted, target):
    """
    predicted (batch_size x num_classes)
    target is labels (batch_size x 1)
    """
    log_likelihood = -np.log(predicted[range(target.shape[0]), target])
    loss = -np.sum(log_likelihood) / target.shape[0]
    # print("Loss: ", np.sum(loss))
    # exit()
    return loss


def get_one_hot_vector(y):
    """
    generating one hot vector for class labels
    """
    y = np.array(y)
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    # y_one_hot = y_one_hot.reshape(-1, 10, 1)
    return y_one_hot


class MLP(object):
    def __init__(self, input_size, load_model_weights, augmented):
        if load_model_weights:
            print("Loading from pretrained model")
            if (augmented):
                read_weights = np.load(load_model_weights + '/augmented-model_weights.npy', allow_pickle='TRUE').item()
            else:
                read_weights = np.load(load_model_weights + '/unaugmented-model_weights.npy',
                                       allow_pickle='TRUE').item()
            self.weights = read_weights['weights']
            self.biases = read_weights['biases']
        else:
            print("Initialising weights and biases")

            self.weights = [np.random.randn(y, x) * 0.1 for x, y in
                            zip(input_size[:-2], input_size[1:-1])] + [
                               np.random.randn(input_size[-1], input_size[-2]) * 0.1]
            self.biases = [np.zeros((x, 1)) for x in input_size[1:-1]] + [np.zeros((input_size[-1], 1))]

        print("Weights shape1: ", self.weights[0].shape)
        print("Weights shape2: ", self.weights[1].shape)
        print("Weights shape3: ", self.weights[2].shape)

        print("Bias shape1: ", self.biases[0].shape)
        print("Bias shape2: ", self.biases[1].shape)
        print("Bias shape3: ", self.biases[2].shape)

    def feedforward(self, x):

        z1 = np.dot(self.weights[0], x) + self.biases[0][:, :x.shape[1]]
        a1 = relu(z1)
        z2 = np.dot(self.weights[1], a1) + self.biases[1][:, :a1.shape[1]]
        a2 = relu(z2)
        z3 = np.dot(self.weights[2], a2) + self.biases[2][:, :a2.shape[1]]
        a3 = softmax(z3.T)

        return z1, a1, z2, a2, z3, a3

    def backpropagation(self, X, y):

        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        z1, a1, z2, a2, z3, a3 = self.feedforward(X)
        y = y.argmax(axis=1)
        loss = cross_entropy_loss(a3, y)
        a3[range(y.shape[0]), y] -= 1
        error = a3

        # For Output Layer
        delta1 = error.T
        delta_b[2] = delta1
        delta_w[2] = np.dot(delta1, a2.T)

        # For Second Hidden Layer
        deriv_relu = relu(z2, deriv=True)
        delta2 = np.dot(self.weights[2].T, delta1) * deriv_relu
        delta_b[1] = delta2
        delta_w[1] = np.dot(delta2, a1.T)

        # For First Hidden Layer
        deriv_relu = relu(z1, deriv=True)
        delta3 = np.dot(self.weights[1].T, delta2) * deriv_relu
        delta_b[0] = delta3
        delta_w[0] = np.dot(delta3, X.T)

        return loss, delta_b, delta_w

    def evaluate(self, X, y):

        count = 0
        preds = np.array([])
        for x, _y in zip(X, y):
            _, _, _, _, _, a3 = self.feedforward(np.array([x]).T)
            preds = np.append(preds, np.argmax(a3))
            if np.argmax(a3) == np.argmax(_y):
                count += 1
        return float(count) / X.shape[0], preds

    def train(self, X, y, X_test, y_test, learning_rate=0.01, epochs=5, batch_size=100):
        history_training_loss, history_training_acc, history_test_acc, history_test_pred = [], [], [], []
        model_weights = {}
        for epoch in range(epochs):
            # Shuffle
            permutation = np.random.permutation(X.shape[0])
            x_train_shuffled = X[permutation]
            y_train_shuffled = y[permutation]

            # same shape as self.biases
            del_b = [np.zeros(b.shape) for b in self.biases]
            # same shape as self.weights
            del_w = [np.zeros(w.shape) for w in self.weights]
            flag = 0
            loss_min = 9999

            for batch_idx in range(0, X.shape[0], batch_size):
                # Selecting batches of the training images
                batch_x = x_train_shuffled[batch_idx:batch_idx + batch_size]
                batch_y = y_train_shuffled[batch_idx:batch_idx + batch_size]

                # Performing Backpropagation
                loss, delta_b, delta_w = self.backpropagation(batch_x.T, batch_y)
                loss = abs(loss)
                # Maintaining the original shape according to batch size when input is less than given batch size
                if delta_b[0].shape[1] < batch_size:
                    # print(delta_b[0].shape)
                    size_left = batch_size - delta_b[0].shape[1]
                    delta_b[0] = np.pad(delta_b[0], ((0, 0), (0, size_left)))

                if delta_b[1].shape[1] < batch_size:
                    # print(delta_b[1].shape)
                    size_left = batch_size - delta_b[1].shape[1]
                    delta_b[1] = np.pad(delta_b[1], ((0, 0), (0, size_left)))

                if delta_b[2].shape[1] < batch_size:
                    # print(delta_b[2].shape)
                    size_left = batch_size - delta_b[2].shape[1]
                    delta_b[2] = np.pad(delta_b[2], ((0, 0), (0, size_left)))

                #     #It will encounter only when there's minimum loss is getting as compared to previous batches
                if loss < loss_min:
                    loss_min = loss
                    del_b = [db + ddb for db, ddb in zip(del_b, delta_b)]
                    del_w = [dw + ddw for dw, ddw in zip(del_w, delta_w)]

            self.weights = [w - (learning_rate / batch_size) * dw for w, dw in zip(self.weights, del_w)]
            self.biases = [b - (learning_rate / batch_size) * db for b, db in zip(self.biases, del_b)]

            model_weights['weights'] = self.weights
            model_weights['biases'] = self.biases

            # Evaluate performance
            train_acc, _ = self.evaluate(X, y)
            test_acc, test_pred = self.evaluate(X_test, y_test)
            history_training_loss.append(loss_min)
            history_training_acc.append(train_acc)
            history_test_acc.append(test_acc)
            history_test_pred.append(test_pred)
            print("Epoch: %d Training loss: %.3f Training accuracy: %.2f Testing Accuracy: %.2f" % (
                epoch, loss_min, train_acc * 100, test_acc * 100))
        return np.array(history_training_acc), np.array(history_training_loss), np.array(history_test_acc), np.array(
            history_test_pred), model_weights


def Model(X_train, y_train, X_test, y_test, model_wt_folder, out_folder, isModelWeightsAvailable=0, epochs=500,
          batch_size=32, learning_rate=0.01, augmented=False):
    inp_feats = 512
    num_neurons_1 = 64
    num_neurons_2 = 64
    num_output = 10
    # batch_size=256
    # learning_rate=0.01
    print("Epochs: ", epochs)
    print("y_train: ", y_train.shape)
    print("batch-size: ", batch_size)
    print("learning_rate: ", learning_rate)
    if isModelWeightsAvailable:
        print("Performance measure on test datasets")
        model = MLP((inp_feats, num_neurons_1, num_neurons_2, num_output), model_wt_folder, augmented)
        acc, prediction = model.evaluate(X_test, y_test)
        print("Testing Accuracy or Performance Measure in percent: %.2f" % (acc * 100))
    else:
        model = MLP((inp_feats, num_neurons_1, num_neurons_2, num_output), '', augmented)
        total_training_acc, total_training_loss, total_testing_acc, total_test_pred, model_weights = model.train(
            X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        # print(model_weights['weights'].shape)
        if (augmented):
            np.save(model_wt_folder + '/augmented-model_weights.npy', model_weights)
        else:
            np.save(model_wt_folder + '/unaugmented-model_weights.npy', model_weights)
        # acc = model.evaluate(X_test, y_test)
        print("Testing Accuracy or Performance Measure in percent: %.2f at epoch: %d" % (
            total_testing_acc[np.argmax(total_training_acc)] * 100, np.argmax(total_training_acc) + 1))
        print("With a loss: %.3f" % (total_training_loss[np.argmax(total_training_acc)]))
        prediction = total_test_pred[np.argmax(total_training_acc)]
        output_dir = './output'
        # Check whether the specified path exists or not
        isExist = os.path.exists(output_dir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(output_dir)
            print("The new directory (output) is created!")
        plt.figure(figsize=(8, 8))
        plt.plot(np.arange(epochs), total_training_acc, label='Total_trainig_acc')
        plt.plot(np.arange(epochs), total_training_loss, label='Total_training_loss')
        plt.plot(np.arange(epochs), total_testing_acc, label='Total_testing_acc')
        plt.xlabel("Epochs")
        plt.ylabel("Training Accuracy and Loss")
        plt.title(
            "Loss/Accuracy vs Epochs - " + "batch_size-" + str(batch_size) + "-learning-rate-" + str(learning_rate))
        plt.legend()
        if (augmented):
            plt.savefig(out_folder + "/augmented-loss-accuracy-graph-" + str(batch_size) + "-" + str(
                learning_rate) + ".jpg")
        else:
            plt.savefig(
                out_folder + "/unaugmented-loss-accuracy-graph-" + str(batch_size) + "-" + str(learning_rate) + ".jpg")

    # Y_test = y_test.argmax(axis=1)
    # print('Accuracy: %.3f' % (accuracy_score(Y_test, prediction)*100))
    # print('Precision: %.3f' % precision_score(Y_test, prediction, average='weighted'))
    # print('Recall: %.3f'% recall_score(Y_test, prediction, average='weighted'))
    # print('F1 score: %.3f'% f1_score(Y_test, prediction, average='weighted'))
    # print('confussion matrix:\n',confusion_matrix(Y_test, prediction))
    return model
