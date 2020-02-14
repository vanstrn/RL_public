import numpy as np

import tensorflow as tf

from utils.utils import store_args

class TrainedNetwork:
    """TrainedNetwork

    Import pre-trained model for simulation only.

    It does not include any training sequence.
    """
    @store_args
    def __init__(
        self,
        model_name,
        input_tensor='global/state:0',
        output_tensor='global/actor/Softmax:0',
        action_space=5,
        sess=None,
        device=None,
        import_scope=None,
        *args,
        **kwargs
    ):
        if import_scope is not None:
            self.input_tensor = import_scope + '/' + input_tensor
            self.output_tensor = import_scope + '/' + output_tensor
        if sess is None:
            self.graph = tf.Graph()
            self.graph.device(device)
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=self.graph)

        self.model_path = model_name

        # Initialize Session and TF graph
        self._initialize_network()

    def get_action(self, input_tensor):
        with self.sess.as_default(), self.sess.graph.as_default():
            feed_dict = {self.state: input_tensor}
            action_prob = self.sess.run(self.action, feed_dict)

        action_out = [np.random.choice(self.action_space, p=prob / sum(prob)) for prob in action_prob]

        return action_out

    def reset_network_weight(self, path=None, step=None):
        if path is not None:
            self.model_path = path

        if step is None:
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            tf.keras.backend.clear_session()
            with self.sess.graph.as_default():
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                else:
                    raise AssertionError
        else:
            full_path = self.model_path + '/ctf_policy.ckpt-' + str(step)
            tf.keras.backend.clear_session()
            with self.sess.graph.as_default():
                self.saver = tf.train.import_meta_graph(
                        full_path+'.meta',
                        clear_devices=True,
                        import_scope=self.import_scope
                    )
                self.saver.restore(self.sess, full_path)

    def _initialize_network(self, verbose=False):
        """reset_network
        Initialize network and TF graph
        """
        def vprint(*args):
            if verbose:
                print(args)

        input_tensor = self.input_tensor
        output_tensor = self.output_tensor

        # Reset the weight to the newest saved weight.
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        vprint(f'path find: {ckpt.model_checkpoint_path}')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            vprint(f'path exist : {ckpt.model_checkpoint_path}')
            with self.sess.graph.as_default():
                self.saver = tf.train.import_meta_graph(
                    ckpt.model_checkpoint_path + '.meta',
                    clear_devices=True,
                    import_scope=self.import_scope
                )
                self.saver.restore(self.sess, ckpt.model_checkpoint_path, )
                vprint([n.name for n in self.sess.graph.as_graph_def().node])

                self.state = self.sess.graph.get_tensor_by_name(input_tensor)

                try:
                    self.action = self.sess.graph.get_operation_by_name(output_tensor)
                except ValueError:
                    self.action = self.sess.graph.get_tensor_by_name(output_tensor)
                    vprint([n.name for n in self.sess.graph.as_graph_def().node])

            vprint('Graph is succesfully loaded.', ckpt.model_checkpoint_path)
            #iuv(self.sess)
        else:
            vprint('Error : Graph is not loaded')
            raise NameError

    def _get_node(self, name):
        try:
            node = self.sess.graph.get_operation_by_name(name)
        except ValueError:
            node = self.sess.graph.get_tensor_by_name(name)
        return node
