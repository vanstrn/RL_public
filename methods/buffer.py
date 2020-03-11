import numpy as np
import random

""" Buffer modules in CtF Project

Classes:
    :Experience_buffer: (in utils.py)

    :Trajectory: buffer Buffer to store the single trajectory roll
        Each trajectory represent the series of MDP tuples for fixed-time
        It is used to represent the single experience by single num_agent.
        The storage is in order of push, and it is stored in 2-D list
        The 'trim' method to divide the trajectory into sub-trajectories.

    :Trajectory_buffer: Advanced buffer used to keep the trajectory correlation between samples
        Advanced version of Experience Buffer.
        The container in format of multidimension list.
        The sampling returns the column of trajectories in numpy list.

Note:
    Use 'from buffer import <class_name>' to use specific method/class
    Please include the docstrings for any method or class to easily reference from Jupyter
    Any pipeline or data manipulation is excluded: they are included in dataModule.py file.

Todo:
    * Experience buffer
    * Improve Trajectory by preventing samples that are done from adding new samples.

"""

def random_batch_sampling(batch_size, epoch, *argv):
    """
    The number of batch is defined by the size // batch_size.
    Number of batch is multiplied by epoch.
    Each sampling is done randomly without replacement.

    It does not use shuffle or removal, which makes it faster to operate with large dataset

    argv contains numpy ndarray of dataset. All array should be equal length.
    """
    lengths = [len(arg) for arg in argv]
    assert len(set(lengths))<=1
    size = lengths[0]
    num_batch = epoch * (size // batch_size)
    for _ in range(num_batch):
        rand_ids = np.random.randint(0, size, batch_size)
        yield tuple(arg[rand_ids] for arg in argv)

def expense_batch_sampling(batch_size, epoch, *argv):
    """
    The number of batch is defined by the size // batch_size.
    Number of batch is multiplied by epoch.
    Each sampling is done without replacement

    It does not use shuffle or removal, which makes it faster to operate with large dataset

    argv contains numpy ndarray of dataset. All array should be equal length.
    """
    lengths = [len(arg) for arg in argv]
    assert len(set(lengths))<=1
    size = lengths[0]
    indices = np.arange(size); random.shuffle(indices)
    start_indices = np.arange(0, size, batch_size)
    for _ in range(epoch):
        for start_index in start_indices:
            end_index = min(start_index+batch_size, size)
            ids = indices[start_index:end_index]
            yield tuple(arg[ids] for arg in argv)


class Trajectory:
    """ Trajectory

    Trajectory of [s0, a, r, s1] (or any other MDP tuples)

    Equivalent to : list [[s0 a r s1]_0, [s0 a r s1]_1, [s0 a r s1]_2, ...]
    Shape of : [None, Depth]
    Each depth must be unstackable
    Does not guarentee the order of push.

    Key Terms:
        depth : number of element in each point along the trajectory.
            ex) [s0, a, r, s1] has depth 4

    Methods:
        __repr__
        __len__ : Return the length of the currently stored trajectory
        is_full : boolean, whether trajectory is full
        append (list)

    Notes:
        - Trajectory is only pushed single node at a time.

    """

    def __init__(self, depth=4):
        # Configuration
        self.depth = depth

        # Initialize Components
        self.buffer = [[] for _ in range(depth)]

    def __repr__(self):
        return f'Trajectory (depth={self.depth}'

    def __len__(self):
        return len(self.buffer[0])

    def __getitem__(self, index):
        return self.buffer[index]

    def __setitem__(self, key, item):
        self.buffer[key] = item

    def append(self, mdp_tup):
        for buf, element in zip(self.buffer, mdp_tup):
            buf.append(element)

    def trim(self, trim_length):
        traj_list = []
        s_, e_ = len(self.buffer[0]) - trim_length, len(self.buffer[0])
        while e_ > 0:
            new_traj = Trajectory(depth=self.depth)
            new_buffer = [buf[max(s_, 0):e_] for buf in self.buffer]
            new_traj.buffer = new_buffer
            traj_list.append(new_traj)
            s_ -= trim_length
            e_ -= trim_length
        return traj_list

    def clear(self):
        self.buffer = [[] for _ in range(self.depth)]

class Trajectory_buffer:
    """Trajectory_buffer

    Buffer for trajectory storage and sampling
    Once the trajectory is pushed, altering trajectory would be impossible. (avoid)

    The shape of the buffer must have form [None, None, depth]
    Each depth must be unstackable, and each unstacked array will have form [None, None]+shape

    Second shape must be consist with others.

    Each trajectory is stored in list.
    At the moment of sampling, the list of individual element is returned in numpy array format.

    Methods:
        __repr__
        __len__ : Return the length of the currently stored number of trajectory
        is_empty : boolean, whether buffer is empty. (length == 0)
        is_full : boolean, whether buffer is full
        append (list)
        extend (list)
        flush : empty, reset, return remaining at once
        sample (int) : sample 'return_size' amount of trajectory

    Notes:
        - The sampling uses random.shuffle() method on separate index=[0,1,2,3,4 ...] array
        - The class is originally written to use for A3C with LSTM network. (Save trajectory in series)

    TODO:
        - Think about better method of handling 2-D trajectory elements
    """

    def __init__(self, depth=4, capacity=256):
        """__init__

        :param capacity: maximum size of the buffer.
        :param return_size: size to return
        """

        # Configuration
        self.depth = depth
        self.capacity = capacity

        # Initialize Components
        self.buffer_size = 0
        self.buffer = [[] for _ in range(self.depth)]

    def __call__(self):
        return self.buffer

    def __repr__(self):
        str = f'Trajectory Buffer(capacity = {self.capacity}, return_size = {self.return_size})'
        return str

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, index):
        return self.buffer[index]

    def __setitem__(self, index, item):
        self.buffer[index] = item

    def is_empty(self):
        return self.buffer_size == 0

    def is_full(self):
        return self.buffer_size == self.capacity

    def append(self, traj):
        for i, elem in enumerate(traj):
            self.buffer[i].append(elem)
        self.buffer_size += 1

    def extend(self, trajs):
        for traj in trajs:
            for i, elem in enumerate(traj):
                self.buffer[i].append(elem)
        self.buffer_size += len(trajs)
        # if len(self.buffer) > self.capacity:
        #     self.buffer = self.buffer[-self.capacity:]
        #     self.buffer_size = len(self.buffer)

    def sample(self, flush=True):
        """sample

        Return in (None,None)+shape
        All returns are in tensor format

        :param flush: True - Emtpy the buffer after sampling
        """
        if flush:
            # Find longest length sequence
            length = 0
            for batch in self.buffer[1]:  # 1 : action array
                length = max(length, len(batch))
            for bid, buf in enumerate(self.buffer):
                for idx, batch in enumerate(buf):
                    batch = np.array(batch)
                    if len(batch) < length:
                        extra_length = length - len(batch)
                        shape = [extra_length] + list(batch.shape[1:])
                        batch = np.append(batch, np.zeros(shape), axis=0)
                    self.buffer[bid][idx] = batch
            ret = tuple(np.array(b) for b in self.buffer)
            self.buffer = [[] for _ in range(self.depth)]
        else:
            raise NotImplementedError
        return ret


class Replay_buffer:
    """Replay_buffer
    Replaybuffer use for storing tuples
    Support returning and shuffling features
    Store in list.
    Order is not guranteed

    Method:
        __init__ (int, int)
        __len__
        add (list)
        add_element (object)
        flush
        empty
        sample (int, bool)
        pop (int, bool)
    """

    def __init__(self, depth=4, buffer_size=5000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.depth = depth

    def __len__(self):
        return len(self.buffer)

    def __call__(self):
        return self.buffer

    def __getitem__(self, index):
        return self.buffer[index]

    def append(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)

    def extend(self, samples):
        if len(self.buffer) + len(samples) >= self.buffer_size:
            random.shuffle(self.buffer)
            self.buffer[0:(len(samples) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(samples)

    def flush(self):
        # Return the remaining buffer and reset.
        batch = self.buffer
        self.buffer = []
        return batch

    def empty(self):
        return len(self.buffer) == 0

    def full(self):
        return len(self.buffer) >= self.buffer_size

    def shuffle(self):
        random.shuffle(self.buffer)

    def pop(self, size, shuffle=False):
        # Pop the first `size` items in order (queue).
        if shuffle:
            self.shuffle()
        batch = self.buffer[:size]
        self.buffer = self.buffer[size:]
        return batch


if __name__ == '__main__':
    print('Debuging')
    a = np.random.randint(5, size=10)
    b = np.random.randint(5, size=10)

    tr = Trajectory(depth=2)
    for t in zip(a, b):
        tr.append(t)
    print(f'Original two list : {a}, {b}')
    print(f'Buffer: {tr.buffer}')
    print(f'Trim by 3 : {[ttr.buffer for ttr in tr.trim(3)]}')

    tr_buf = Trajectory_buffer(depth=2)
    tr_buf.extend(tr.trim(serial_length=2))
    s1, s2 = tr_buf.sample()
    print(f'Trim by 2 : {[ttr.buffer for ttr in tr.trim(2)]}')
    print(f's1 : {s1}')
    print(f's2 : {s2}')
