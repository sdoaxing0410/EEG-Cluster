#! /usr/bin/env python3.7
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold


def MDS(data, y, y_refer):
    X = np.array(data)
    mds = manifold.MDS()
    X = mds.fit_transform(X)
    print('stress(n_components = 2):%s' % mds.stress_)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    for label in np.unique(y):
        position = y == label
        ax1.scatter(X[position, 0], X[position, 1])
    ax1.set_xlabel('X[0]')
    ax1.set_ylabel('X[1]')
    ax1.set_title('cluster result')

    ax2 = fig.add_subplot(212)
    for label in np.unique(y_refer):
        position = y_refer == label
        ax2.scatter(X[position, 0], X[position, 1])
    ax2.set_xlabel('X[0]')
    ax2.set_ylabel('X[1]')
    ax2.set_title('reference label')
    plt.show()




