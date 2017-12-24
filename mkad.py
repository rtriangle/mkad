#!/usr/bin/env python
"""
Copyright (c) 2017 Daniil Korbut
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import rbf_kernel, poly_kernel, linear_kernel
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import OneClassSVM
from collections import Counter
from pyts.transformation import StandardScaler
from pyts.transformation import PAA
from pyts.transformation import SAX


class MultipleKernelAnomalyDetector:
    """
        Multiple Kernel anomaly-detection method implementation
    """

    def __init__(self,
                 nu=0.25,
                 tol=1e-3,
                 degree=3,
                 kernel='lcs',
                 sax_size=5,
                 quantiles='gaussian'
                 ):
        """
            Constructor accepts some args for sklearn.svm.OneClassSVM and SAX inside.
            Default params are choosen as the most appropriate for flight-anomaly-detection problem
            according the original article.
        """
        self.nu = nu
        self.tol = tol
        self.degree = degree
        self.kernel = kernel
        self.stand_scaler = StandardScaler(epsilon=1e-2)
        self.paa = PAA(window_size=None, output_size=8, overlapping=True)
        self.sax = SAX(n_bins=sax_size, quantiles=quantiles)

    def compute_matrix_of_equals(self, sequence1, sequence2):
        """
            Computes matrix, where at (i, j) coordinate is the lcs for sequence1[:i+1] and sequence2[:j+1]
        """
        lengths = np.zeros((len(sequence1) + 1, len(sequence2) + 1))
        for i, element1 in enumerate(sequence1):
            for j, element2 in enumerate(sequence2):
                if element1 == element2:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
        return lengths

    def lcs(self, sequence1, sequence2):
        """
            Computes largest common subsequence of sequence1 and sequence2
        """
        lengths = self.compute_matrix_of_equals(sequence1, sequence2)
        result = ""
        i, j = len(sequence1), len(sequence2)
        while i != 0 and j != 0:
            if lengths[i][j] == lengths[i - 1][j]:
                i -= 1
            elif lengths[i][j] == lengths[i][j - 1]:
                j -= 1
            else:
                assert sequence1[i - 1] == sequence2[j - 1]
                result = sequence1[i - 1] + result
                i -= 1
                j -= 1
        return result

    def nlcs(self, sequence1, sequence2):
        """
            Computes normalized common subsequence of sequence1 and sequence2
        """
        return len(self.lcs(sequence1, sequence2)) / (len(sequence1) * len(sequence2)) ** 0.5

    def get_sax(self, sequence):
        sequence = np.reshape(sequence, (1, len(sequence)))
        return self.sax.transform(self.paa.transform(self.stand_scaler.transform(sequence)))[0]
    
    def lcs_kernel_function(self, x1, x2):
        """
             LCS - kernel for Multiple Kernel Anomaly Detector
        """
        res = np.zeros((x1.shape[0], x2.shape[0]))
        for ind1 in range(x1.shape[0]):
            for ind2 in range(x2.shape[0]):
                for i in range(0, len(x1[ind1]), self.x_shape[-1]):
                    res[ind1][ind2] += self.nlcs(self.get_sax(x1[ind1][i:i+self.x_shape[-1]]),
                                          self.get_sax(x2[ind2][i:i+self.x_shape[-1]]))
        print(res.shape)
        return res

    def __transformation(self, x):
        """
            Transforms X from 3D to 2D array for OneClassSVM
        """
        return x.transpose(0, 1, 2).reshape(x.shape[0], -1)
    
    def fit(self, x):
        """
            With lcs kernel X must have shape (n, d, l),
            where n - number of samples, d - number of dimensions, l - feature length.
            With rbf kernel X must have shape (n, l)
            where n - number of samples, l - feature length.
        """
        self.x_shape = x.shape
        if self.kernel == 'lcs':
            x_transformed = self.__transformation(x)
            f = lambda x, y: self.lcs_kernel_function(x,y) 
            self.one_class_svm = OneClassSVM(kernel=f)
            self.one_class_svm.fit(x_transformed)
        else:
            x_transformed = x
            self.one_class_svm = OneClassSVM(kernel='rbf')
            self.one_class_svm.fit(x_transformed)

    def predict(self, x):
        '''
            With lcs kernel X must have shape (n, d, l),
            where n - number of samples, d - number of dimensions, l - feature length.
            With rbf kernel X must have shape (n, l)
            where n - number of samples, l - feature length.
            Function returns y-array with +1;-1
        '''
        if len(x.shape) > 2:
            x = self.__transformation(x)
        return self.one_class_svm.predict(x)
