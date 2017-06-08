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
        
        self.grad = np.zeros(len(self.trainingSet.input[0]))

	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for j in range(self.epochs):
        	targets = []
        	for tar in self.trainingSet.label:
        		if(tar == 1):
        			targets.append(True)
        		else:
        			targets.append(False)        	
        	targets = np.array(self.trainingSet.label)
        	inputs = np.array(self.trainingSet.input)
        	outputs = self.evaluate(inputs)
        	
        	index = 0
        	
        	for instance in self.trainingSet.input:
        		output = self.classify(instance)
        		error = targets[index] - output
        		self.grad = self.grad + error * instance
        		self.updateWeights(self.grad)
        		index = index + 1
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
        x = np.dot(self.weight,testInstance)        
        if((1 / (1 + np.exp(-x))) > 0.5):
        	return True
        else:
        	return False
        #pass

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
    	  self.weight = self.weight + self.learningRate * self.grad
    	  #print(self.grad)
        #pass

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
