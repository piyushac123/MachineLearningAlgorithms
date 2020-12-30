#import pandas as pd
import numpy as np
from numpy.linalg import eig
numpy_array = np.genfromtxt("Dataset3.csv", delimiter=",")
#Centering
mean_na = numpy_array.mean(0)
print("Mean :")
print(mean_na)
for i in range(len(numpy_array)):
    numpy_array[i] -= mean_na
#Covariance C = (numpy_array.T.dot(numpy_array))/n
C = np.cov(numpy_array.T)
print("Covariance :")
print(C)
e_val,e_vec = eig(C)
print("\nEigen values :")
print(e_val)
print("\nEigen vectors :")
print(e_vec)
#W = e_vec[np.where(e_val == np.amax(e_val))]
W1 = []
print("\nVariance :")
for i in e_vec:
    W = i
    W1.append(W.dot(C.dot(W.T)))
print(W1)
