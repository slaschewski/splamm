
import numpy as np

from util.loss_functions import CrossEntropyError, BinaryCrossEntropyError, SumSquaredError, MeanSquaredError, DifferentError, AbsoluteError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        #self.cost = cost

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128, 
                           None, inputActivation, False))

        self.layers.append(LogisticLayer(128,100, None, "sigmoid", False))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(100, 10, 
                           None, outputActivation, True))

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        input_next = inp
        for layer in self.layers:
            lastLayerOutput = layer.forward(input_next)
            input_next = np.insert(lastLayerOutput,0,1)

        return lastLayerOutput

        
    def _compute_error(self, target, output):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        error = np.zeros(self.layers[-1].nOut)
        for i in range(self.layers[-1].nOut):
            error[i] = self.loss.calculateError(target[i],output[i])
        return error
        
    
    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """
        for layer in self.layers:
            layer.updateWeights(learningRate)
        
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for epoch in range(self.epochs):
            for img, label in zip(self.trainingSet.input,
                                self.trainingSet.label):

                label_vec = np.zeros([10])
                label_vec[label] = 1

                label = label_vec
                #print("label " + str(label))

                # Use LogisticLayer to do the job
                # Feed it with inputs
                output = self._feed_forward(img)
                #print("out " + str(output))

                # Do a forward pass to calculate the output and the error
                error = self._compute_error(label, output)

                nextWeights = np.ones(self.layers[-1].nOut)
                nextDev = error
                #print("--")
                for layer in reversed(self.layers):
                    #print("Dev " + str(nextDev))
                    nextDev = layer.computeDerivative(nextDev, nextWeights)
                    nextWeights = np.transpose(layer.weights[1:])
                    #print("weights " + str(nextWeights))

                # Update weights in the online learning fashion
                self._update_weights(self.learningRate)

                '''if verbose:
                    accuracy = accuracy_score(self.validationSet.label,
                                              self._feed_forward(self.validationSet))
                    # Record the performance of each epoch for later usages
                    # e.g. plotting, reporting..
                    self.performances.append(accuracy)
                    print("Accuracy on validation: {0:.2f}%"
                          .format(accuracy * 100))
                    print("-----------------------------")'''

            print(epoch)


            '''Break condition'''
            if False:
                break
                





    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        outp = self._feed_forward(test_instance)
        print(outp)
        return np.argmax(outp)
        

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
