# pylint: disable=redefined-outer-name, unreachable, missing-function-docstring
"""
Image Processing Project 0: Starter Code
"""

# Ziming Wang
# CS 6640 Image Processing
# Prof.Tolga Tasdizen
# 9/2/2022

import numpy as np

# Question 1.a:
def numpy_vec_op():
    # TODO: generate a numpy vector containing 0-4 using numpy function arange()
    ##  ***************** Your code starts here ***************** ##

    generate_vec = np.arange(0, 5)

    #raise NotImplementedError

    ##  ***************** Your code ends here ***************** ##

    x = np.array([4, 5, 6, 7, 8])

    # TODO: perform a dot product with 'x', store it in 'result'

    ##  ***************** Your code starts here ***************** ##

    result = np.dot(generate_vec, x)

    #raise NotImplementedError

    ##  ***************** Your code ends here ***************** ##

    if result != 70.0:
        print("\033[91mIncorrect!\033[00m")
    else:
        print("\033[92mCorrect!\033[00m")


# Question 1.b:
def numpy_mat_op():
    # TODO: generate a numpy matrix [[0, 1], [2, 3]] using numpy function arange() and reshape()
    ##  ***************** Your code starts here ***************** ##

    generate_mat = np.arange(0, 4).reshape(2, 2)

    #raise NotImplementedError

    ##  ***************** Your code ends here ***************** ##

    A = np.array([[5, 6], [7, 8]])

    # TODO: perform a dot product with 'A'
    ##  ***************** Your code starts here ***************** ##

    result = np.dot(generate_mat, A)
    #raise NotImplementedError

    ##  ***************** Your code ends here ***************** ##

    if np.sum(result - np.array([[7, 8], [31, 36]])) != 0:
        print("\033[91mIncorrect!\033[00m")
    else:
        print("\033[92mCorrect!\033[00m")


# Question 1.c:
def numpy_3d_op():
    # TODO: create a 10x10x3 matrix (think of this as an RGB image)
    ##  ***************** Your code starts here ***************** ##

    mat = np.arange(0, 300).reshape(10, 10, 3)

    # raise NotImplementedError

    ##  ***************** Your code ends here ***************** ##

    weights = [0.3, 0.6, 0.1]

    # TODO: Compute the dot-product of the 10x10x3 matrix and 'weights' to get a 10x10 matrix
    ##  ***************** Your code starts here ***************** ##

    result = np.dot(mat, weights)
    # raise NotImplementedError

    ##  ***************** Your code ends here ***************** ##
    # Notice that you just took your 3-color "image" and made it 1 color!

    if result.shape != (10, 10):
        print("\033[91mIncorrect!\033[00m")
    else:
        print("\033[92mCorrect!\033[00m")


if __name__ == "__main__":
    print("Question 1.a: ", end="")
    numpy_vec_op()
    print("Question 1.b: ", end="")
    numpy_mat_op()
    print("Question 1.c: ", end="")
    numpy_3d_op()
