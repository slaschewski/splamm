# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from report.evaluator import Evaluator

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, 
                                    learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/100

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for i in range(self.epochs):
            targets = []
            for target in self.trainingSet.label:
                if target == 1:
                    targets.append(True)
                else:
                    targets.append(False)

            targets = np.array(targets)

            inputs = np.array(self.trainingSet.input)
            outputs = self.evaluate(inputs)

            errors = []

            for j in range(len(targets)):
                if outputs[j] == targets[j]:
                    errors.append(0)
                elif outputs[j]:
                    errors.append(1)
                else:
                    errors.append(-1)


            self.updateWeights(inputs, errors)

            if verbose:
                validation = self.evaluate(self.validationSet)
                evaluator = Evaluator()
                evaluator.printAccuracy(self.validationSet, validation)


    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.fire(testInstance)


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

    def updateWeights(self, input, error):

        for i in range(len(self.weight)):
            sum = 0
            for j in range(len(error)):
                sum += error[j] * input[j][i]
            #print sum
            self.weight[i] -= self.learningRate * sum


         
    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))
