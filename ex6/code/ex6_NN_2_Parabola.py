import sys
from trainer import Trainer
import numpy as np


def parabolaData():
    X_train = np.load('../data/train_inputs.npy').T
    y_train = np.load('../data/train_targets.npy')
    X_test = np.load('../data/validation_inputs.npy').T
    y_test = np.load('../data/validation_targets.npy')

    nTrainSamples = X_train.shape[0]
    print("Total number of training examples: {}".format(nTrainSamples))
    batch_size = 20
    trainer = Trainer(input_size=X_train.shape, batch_size=batch_size, lr=0.1, weight_decay=0.0)

    trainer.train(X_train.astype(np.float32), y_train.astype(np.float32), X_test.astype(np.float32),
                  y_test.astype(np.float32),
                  num_of_epochs_total=500, batch_size=batch_size, output_folder='output/')


if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("Neural network parabola example!")
    parabolaData()
    print("Done!")

# validation accuracy after 500 epochs of training: 78.5