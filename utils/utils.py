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
