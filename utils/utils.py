# Module contains any methods, class, parameters, etc that is related to logging the trainig

import numpy as np
import tensorflow as tf


def InitializeVariables(sess):
    """
    Initializes uninitialized variables

    Parameters
    ----------------
    sess : [tensorflow.Session()]
    """
    from itertools import compress
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))
    non_init_length = len(not_initialized_vars)

    print(f'{non_init_length} number of non-initialized variables found.')

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
        print('Initialized all non-initialized variables')

class MovingAverage:
    """MovingAverage
    Container that only store give size of element, and store moving average.
    Queue structure of container.
    """

    def __init__(self, size):
        """__init__

        :param size: number of element that will be stored in he container
        """
        from collections import deque
        self.average = 0.0
        self.size = size

        self.queue = deque(maxlen=size)

    def __call__(self):
        """__call__"""
        return self.average

    def tolist(self):
        """tolist
        Return the elements in the container in (list) structure
        """
        return list(self.queue)

    def extend(self, l: list):
        """extend

        Similar to list.extend

        :param l (list): list of number that will be extended in the deque
        """
        # Append list of numbers
        self.queue.extend(l)
        self.size = len(self.queue)
        self.average = sum(self.queue) / self.size

    def append(self, n):
        """append

        Element-wise appending in the container

        :param n: number that will be appended on the container.
        """
        s = len(self.queue)
        if s == self.size:
            self.average = ((self.average * self.size) - self.queue[0] + n) / self.size
        else:
            self.average = (self.average * s + n) / (s + 1)
        self.queue.append(n)

    def clear(self):
        """clear
        reset the container
        """
        self.average = 0.0
        self.queue.clear()
