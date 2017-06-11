# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

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
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """


        learned = False
        iteration = 0

        from util.loss_functions import DifferentError
        loss = DifferentError()


        # grad = [0]
        grad = np.zeros(len(self.trainingSet.input[0]))


        # Train for some epochs if the error is not 0
        while not learned:
            # x ist ein Bild bestehend aus einem Label (erster Eintrag) und 784 Pixeln
            # t ist das Zielergebnis von x (überprüfbar mit dem Label)
            # o ist der tatsächliche Ergebnis von x
            # w ist der Gewichtsvektor
            # Als Aktivierungsfunktion verwenden wir die Sigmoid Funktion
            # Das Training wird dann beendet, sobald das Fehlerkriterium konvergiert

            totalError = 0

            output = []
            labels = self.trainingSet.label
            inputs = self.trainingSet.input

            # iteriere für jede Instanz im Trainingsset x € X
            for input, label in zip(inputs,
                                    labels):
                # Ermittle Ox = sig(w*x)
                output.append(self.fire(input))

            # Ermittle Fehler AE = tx - ox
            error = loss.calculateError(np.array(labels), np.array(output))

            print error
            for e ,input in zip(error, inputs):
                grad += np.multiply( input, e)
                # Update grad = grad + error * x


            #print "Error: " + str(error) + " Grad: " + str(grad)

            # update w: w <- w + n*grad
            self.updateWeights(grad)


            iteration += 1
            totalError = error.sum()

            if verbose:
                logging.info("Epoch: %i; Error: %i", iteration, totalError)

            if totalError == 0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True

        pass
        
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

    def updateWeights(self, grad):
        self.weight += self.learningRate * grad
        pass

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
