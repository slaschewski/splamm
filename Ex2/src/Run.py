#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator
import numpy as np

def main():


    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    # myStupidClassifier = StupidRecognizer(data.trainingSet,
    #                                       data.validationSet,
    #                                       data.testSet)
                                          
    # myPerceptronClassifier = Perceptron(data.trainingSet,
    #                                     data.validationSet,
    #                                     data.testSet,
    #                                     learningRate=0.005,
    #                                     epochs=30)

    myLogisticRegressionClassifier = LogisticRegression(data.trainingSet, data.validationSet, data.testSet, learningRate=0.005, epochs=30 )





    # Train the classifiers
    print("=========================")
    print("Training..")

    # print("\nStupid Classifier started training..")
    # myStupidClassifier.train()
    # print("Done..")

    # print("\nPerceptron started training..")
    # myPerceptronClassifier.train()
    # print("Done..")
    
    print("\nLogisticRegression started training..")
    myLogisticRegressionClassifier.train()
    print("Done..")




    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    # stupidPred = myStupidClassifier.evaluate()
    # perceptronPred = myPerceptronClassifier.evaluate()

    logisticPred = myLogisticRegressionClassifier.evaluate()


    # Report the result
    print("=========================")
    evaluator = Evaluator()

    # print("Result of the stupid recognizer:")
    # # evaluator.printComparison(data.testSet, stupidPred)
    # evaluator.printAccuracy(data.testSet, stupidPred)

    # print("\nResult of the Perceptron recognizer:")
    # # evaluator.printComparison(data.testSet, perceptronPred)
    # evaluator.printAccuracy(data.testSet, perceptronPred)

    print("\nResult of the Logistic Regression recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, logisticPred)

if __name__ == '__main__':
    main()
