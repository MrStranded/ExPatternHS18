import os
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import mlp
import torch.nn as nn
import torch as tc
from torch.autograd import Variable
import matplotlib.pyplot as plt

use_cuda = tc.cuda.is_available()
plt.ioff()


class Trainer(nn.Module):
    '''
    Neural network trainer class
    '''

    def __init__(self, input_size, batch_size, lr, weight_decay):
        '''
        :param input_size:
        :param batch_size:
        :param lr:
        :param weight_decay:
        '''
        super(Trainer, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_mlp = mlp.MLP(self.input_size)
        self.optimizer_model = tc.optim.Adagrad(self.model_mlp.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion_model = nn.CrossEntropyLoss()

    def train(self, X_train, y_train, X_test, y_test, num_of_epochs_total=1000, batch_size=32, output_folder=''):
        '''
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param num_of_epochs_total:
        :param batch_size:
        :param output_folder:
        :return:
        '''
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        prediction_accuracy = np.ones(num_of_epochs_total) * (-1)
        validation_loss = np.ones(num_of_epochs_total) * (-1)
        train_loss = np.ones(num_of_epochs_total) * (-1)
        # instantiate progress bar
        for epoch in range(num_of_epochs_total):
            # Training
            train_loss[epoch] = self.train_model(X_train, y_train)

            # Testing
            prediction_accuracy[epoch], validation_loss[epoch] = self.test_model(X_test, y_test)
            # Plot Loss and Decision Function
            if np.mod(epoch, 10) == 0:
                grid_xlim = [0, 1]
                grid_ylim = [0, 1]
                self.plot_decision_function(X_train, y_train, grid_xlim, grid_ylim, output_folder + 'tmp_' + str(epoch))
                self.plot_loss(train_loss, validation_loss, output_folder + 'loss')
                print('Validation Accuracy epoch [%.4d/%d] %f' % (
                    epoch, num_of_epochs_total, prediction_accuracy[epoch]))

    def train_model(self, X_train, y_train):
        '''
        :param X_train:
        :param y_train:
        :return:
        '''
        self.model_mlp.train()
        train_loss = 0
        num_batches = int(X_train.shape[0] / self.batch_size)

        for batch_idx in range(0, num_batches):
            self.optimizer_model.zero_grad()
            # get data
            slice = self.get_ith_batch_ixs(batch_idx, X_train.shape[0], self.batch_size)
            batch_data = X_train[slice, :]
            inputs = tc.from_numpy(batch_data)
            targets = tc.t(tc.from_numpy(y_train[:, slice]))
            targets = tc.reshape(targets, (targets.shape[0],))
            inputs, targets = Variable(inputs), Variable(targets)
            # forward pass through network
            y_hat = self.model_mlp.forward(inputs)
            _, predictions = tc.max(y_hat, 1)
            # compute loss
            loss = self.criterion_model(y_hat, targets.type(tc.LongTensor))
            loss.backward()
            # make gradient update step
            self.optimizer_model.step()
            # keep track of training error
            train_loss += loss.item()
        return (train_loss / num_batches)

    def test_model(self, X_evaluate, y_evaluate):
        '''
        :param X_evaluate:
        :param y_evaluate:
        :return:
        '''
        self.model_mlp.eval()
        correct = 0
        total = 0
        samples_to_collect_cnt = 0
        num_batches = int(X_evaluate.shape[0] / self.batch_size)
        test_loss = 0

        for batch_idx in range(0, num_batches):
            slice = self.get_ith_batch_ixs(batch_idx, X_evaluate.shape[0], self.batch_size)
            batch_data = X_evaluate[slice, :]
            inputs = tc.from_numpy(batch_data)
            targets = tc.from_numpy(y_evaluate[:, slice])
            inputs, targets = Variable(inputs), Variable(targets)
            targets = tc.reshape(targets, (targets.shape[1],))
            y_hat = self.model_mlp.forward(inputs)
            _, predictions = tc.max(y_hat, 1)

            loss = self.criterion_model(y_hat.type(tc.FloatTensor), targets.type(tc.LongTensor))
            test_loss += loss.item()

            total += targets.size(0)
            predictions = np.reshape(predictions, [predictions.shape[0], ])
            correct += predictions.eq(targets.type(tc.LongTensor)).sum().numpy()

        acc = 100. * correct / total
        test_loss = (test_loss / num_batches)
        return acc, test_loss

    def evaluate_model(self, net, X_evaluate, y_evaluate, batch_size, collect_predictions, num_samples_to_collect):
        '''
        :param net:
        :param X_evaluate:
        :param y_evaluate:
        :param batch_size:
        :param collect_predictions:
        :param num_samples_to_collect:
        :return:
        '''
        net.eval()
        correct = 0
        total = 0
        samples_to_collect_cnt = 0
        num_batches = X_evaluate.shape[0] / batch_size
        iterations_to_save = np.floor(num_batches / num_samples_to_collect).astype(np.int)
        predictions_collected = np.zeros([(iterations_to_save - 1) * batch_size])
        test_loss = 0

        for batch_idx in range(0, iterations_to_save):  # , (inputs, targets) in enumerate(trainloader):
            slice = self.get_ith_batch_ixs(batch_idx, X_evaluate.shape[0], batch_size)
            batch_data = X_evaluate[slice, :]
            inputs = tc.from_numpy(batch_data)
            targets = tc.from_numpy(y_evaluate[:, slice])
            inputs, targets = Variable(inputs), Variable(targets)
            targets = tc.reshape(targets, (targets.shape[1],))
            y_hat, features = net.forward(inputs)
            _, predictions = tc.max(y_hat, 1)

            loss = self.criterion_model(y_hat.type(tc.FloatTensor), targets.type(tc.LongTensor))
            test_loss += loss.item()

            total += targets.size(0)
            predictions = np.reshape(predictions, [predictions.shape[0], ])
            correct += predictions.eq(targets.type(tc.LongTensor)).sum().numpy()
            if collect_predictions:
                predictions_collected[samples_to_collect_cnt * batch_size:(
                                                                          samples_to_collect_cnt + 1) * batch_size] = predictions.detach().numpy()

        acc = 100. * correct / total

        if collect_predictions:
            return acc, predictions_collected
        else:
            print('MLP_MODEL Accuracy:', acc)
            return acc, (test_loss / iterations_to_save)

    def plot_decision_function(self, X_train, y_train, grid_xlim, grid_ylim, save_path=None):
        '''
        :param X_train:
        :param y_train:
        :param grid_xlim:
        :param grid_ylim:
        :param save_path:
        :return:
        '''
        xx, yy = np.meshgrid(np.arange(grid_xlim[0], grid_xlim[1], 0.01),
                             np.arange(grid_ylim[0], grid_ylim[1], 0.01))
        data_numpy = np.c_[xx.ravel(), yy.ravel()]
        data = tc.from_numpy(data_numpy).type(tc.FloatTensor)

        # PLOT DECISION FUNCTION
        pred = self.model_mlp(data)
        _, predictions = tc.max(pred, 1)
        predictions = predictions.cpu().numpy()

        Z = np.rint(predictions)  # for plotting the contour

        # PLOT DECISION ON TRAINING DATA
        plt.figure(figsize=(5, 5))
        plt.ylim(grid_ylim)
        plt.xlim(grid_xlim)
        tensor_x_train = tc.from_numpy(X_train).type(tc.FloatTensor)
        pred_train = self.model_mlp(tensor_x_train)
        _, predictions_train = tc.max(pred_train, 1)
        predictions_train = predictions_train.cpu().numpy()
        X_train = X_train.T
        plt.plot(
            [
                X_train[0, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 0 and predictions_train[i] == 0
            ],
            [
                X_train[1, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 0 and predictions_train[i] == 0
            ],
            'o', color='orange', label='true negatives'
        )
        plt.plot(
            [
                X_train[0, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 1 and predictions_train[i] == 1
            ],
            [
                X_train[1, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 1 and predictions_train[i] == 1
            ],
            'o', color='red', label='true positives'
        )
        plt.plot(
            [
                X_train[0, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 0 and predictions_train[i] == 1
            ],
            [
                X_train[1, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 0 and predictions_train[i] == 1
            ],
            'o', color='blue', label='false positives'
        )
        plt.plot(
            [
                X_train[0, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 1 and predictions_train[i] == 0
            ],
            [
                X_train[1, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 1 and predictions_train[i] == 0
            ],
            'o', color='green', label='false negatives'
        )
        if np.sum(Z) > 0:
            plt.contour(xx, yy, Z.reshape(xx.shape))
        plt.show() if save_path is None else plt.savefig(save_path + '_data')
        plt.close()

    def plot_loss(self, train_loss, val_loss, save_path):
        '''
        :param train_loss:
        :param val_loss:
        :param save_path:
        :return:
        '''
        plt.ioff()
        boo = (train_loss != -1)
        train_loss = train_loss[boo]
        val_loss = val_loss[boo]
        plt.figure(figsize=(5, 5))
        max_y = np.max(train_loss)
        if np.isnan(max_y) | np.isinf(max_y):
            print('')
        plt.ylim([0, 1])
        plt.xlim([0, train_loss.shape[0]])
        x = range(0, train_loss.shape[0])
        line1, = plt.plot(x, train_loss, label='train')
        line2, = plt.plot(x, val_loss, label='validation')
        plt.legend(handles=[line1, line2])
        plt.savefig(save_path)
        plt.close()

    def get_ith_batch_ixs(self, i, num_data, batch_size):
        '''
        Split data into minibatches.
        :param i: integer - iteration index
        :param num_data: integer - number of data points
        :param batch_size: integer - number of data points in a batch
        :return: slice object
        '''
        num_minibatches = num_data / batch_size + ((num_data % batch_size) > 0)
        i = i % num_minibatches
        start = int(i * batch_size)
        stop = int(start + batch_size)
        return slice(start, stop)
