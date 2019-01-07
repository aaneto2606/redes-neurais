#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()
entradas = iris.data
saidas = iris.target

redeNeural = MLPClassifier(verbose = True, 
                           max_iter=1000,
                           tol = 0.00001,
                           activation ='logistic',
                           learning_rate_init = 0.001)
redeNeural.fit(entradas, saidas)
redeNeural.predict([[5, 7.2, 5.1, 2.2]])

redeNeural.predict([[3, 5.2, 7.1, 1.8]])

redeNeural.predict([[5.5, 2.3, 4, 1.3]])

redeNeural.predict([[5.8, 2.3, 1.4, 1.5]])


evaluation = datasets.load_boston()
entradas1 = evaluation.data
saidas1 = evaluation.target

"""
redeNeural = MLPClassifier(verbose = True, 
                           max_iter=1000,
                           tol = 0.00001,
                           activation ='logistic',
                           learning_rate_init = 0.001)

redeNeural.fit(entradas1, saidas1)
"""