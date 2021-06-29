import tensorflow as tf
import sonnet as snt

from deeptrack.models.utils import normalize_dwi_tf

class Seq2SeqModel(object):
    
    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf
        
    def build_graph(self):
        
        self.dwi = tf.placeholder(tf.float32, shape=[None, None, 33])
        self.tract = tf.placeholder(tf.float32, shape=[None, None, 3])
        self.length = tf.placeholder(tf.int64, shape=[None])
        
        
        self.encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._conf['encoder_lstm_units'],
                                            state_is_tuple=True)
        
        self.decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._conf['decoder_lstm_units'])
        
        encoder_outputs, encoder_last_state = self.encode(self.dwi)
        
        
        
        
    def encode(self, inputs):
        outputs, last_states = tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                                           dtype=tf.float32,
                                                           sequence_length = self.length,
                                                           inputs=inputs)
        
        return encoder_output, encoder_last_states
        
    def decode(self, decoder_inputs, encoder_last_state):
        
        decoder_outputs, decoder_last_state = tf.nn.dynamic_rnn(self.decoder_cell,
                                                                 decoder_inputs,
                                                                 initial_state=encoder_last_state,
                                                                 dtype=tf.float32,
                                                                 time_major=True
            )
        
        return decoder_outputs, decoder_last_state
    
                
class SimpleRNNModel(object):
    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf
        
        self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self._conf['num_units'])
        
        self.MAX_GRAD_NORM = 5.0
        self.learning_rate = 1e-3
        
        
        
    def build_graph(self):
        with self._graph.as_default():
            self.dwi = tf.placeholder(tf.float32, shape=[None, None, 33])
            self.tract = tf.placeholder(tf.float32, shape=[None, None, 3])
            self.length = tf.placeholder(tf.int64, shape=[None])
            
            if self._conf['normalize_dwi']:
                self.dwi = normalize_dwi_tf(self.dwi)
            
            self.initial_state = self.rnn_cell.zero_state(self._conf['batch_size'], dtype=tf.float32)
            
            self.outputs, self.last_state = tf.nn.dynamic_rnn(self.rnn_cell, 
                                                    self.dwi, 
                                                    initial_state=self.initial_state,
                                                    dtype=tf.float32)
            
            self.predictions = tf.layers.dense(self.outputs, self._conf['output_size'])
            
            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.tract, predictions=self.predictions))
            
            
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.MAX_GRAD_NORM)
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
            
            
            with tf.variable_scope("performance"):
                self.loss_summary = tf.summary.scalar('loss', self.loss)
                
                self.merged_metric = tf.summary.merge_all()
            
        return self._graph

    def run(self, sess, feed_dict):
        sess.run(self.train_op, feed_dict=feed_dict)
        
        
class MultiRNNModel(object):
    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf
        
        self.MAX_GRAD_NORM = 5.0
        self.learning_rate = 1e-3
        
    def get_a_cell(self, name=None):
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self._conf['num_units'])
            drop = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, 
                                                output_keep_prob=self._conf['keep_prob'],
                                                name=name)
            return drop
        
    def build_graph(self):
        with self._graph.as_default():
            self.dwi = tf.placeholder(tf.float32, shape=[None, None, 33])
            self.tract = tf.placeholder(tf.float32, shape=[None, None, 3])
            self.length = tf.placeholder(tf.int64, shape=[None])
            
            if self._conf['normalize_dwi']:
                self.dwi = normalize_dwi_tf(self.dwi)
            
            self.cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.get_a_cell(lstm_layer_id) for lstm_layer_id in range(self._conf['rnn_depth'])]
            )
            
            self.initial_state = self.cell.zero_state(self._conf['batch_size'], dtype=tf.float32)
            
            self.outputs, self.last_state = tf.nn.dynamic_rnn(self.cell, 
                                                    self.dwi, 
                                                    initial_state=self.initial_state,
                                                    dtype=tf.float32)
            
            self.predictions = tf.layers.dense(self.outputs, self._conf['output_size'])
            
            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.tract, predictions=self.predictions))
            
            
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.MAX_GRAD_NORM)
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
            
            
            with tf.variable_scope("performance"):
                self.loss_summary = tf.summary.scalar('loss', self.loss)
                
                self.merged_metric = tf.summary.merge_all()
            
        return self._graph

    def run(self, sess, feed_dict):
        sess.run(self.train_op, feed_dict=feed_dict)
        

                
class SimpleLSTMModel(object):
    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf
        
        self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self._conf['num_units'])
        
        self.MAX_GRAD_NORM = 5.0
        self.learning_rate = 1e-3
        
        
        
    def build_graph(self):
        with self._graph.as_default():
            self.dwi = tf.placeholder(tf.float32, shape=[None, None, 33])
            self.tract = tf.placeholder(tf.float32, shape=[None, None, 3])
            self.length = tf.placeholder(tf.int64, shape=[None])
            
            if self._conf['normalize_dwi']:
                self.dwi = normalize_dwi_tf(self.dwi)
            
            self.initial_state = self.rnn_cell.zero_state(self._conf['batch_size'], dtype=tf.float32)
            
            self.outputs, self.last_state = tf.nn.dynamic_rnn(self.rnn_cell, 
                                                    self.dwi, 
                                                    initial_state=self.initial_state,
                                                    dtype=tf.float32)
            
            self.predictions = tf.layers.dense(self.outputs, self._conf['output_size'])
            
            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.tract, predictions=self.predictions))
            
            
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.MAX_GRAD_NORM)
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
            
            
            with tf.variable_scope("performance"):
                self.loss_summary = tf.summary.scalar('loss', self.loss)
                
                self.merged_metric = tf.summary.merge_all()
            
        return self._graph

    def run(self, sess, feed_dict):
        sess.run(self.train_op, feed_dict=feed_dict)

            

class MultiLSTMModel(object):
    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf
        
        self.MAX_GRAD_NORM = 5.0
        self.learning_rate = 1e-3
        
    def get_a_cell(self):
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self._conf['num_units'])

            drop = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self._conf['keep_prob'])
            return drop
        
    def build_graph(self):
        with self._graph.as_default():
            self.dwi = tf.placeholder(tf.float32, shape=[None, None, 33], name='dwi')
            self.tract = tf.placeholder(tf.float32, shape=[None, None, 3], name='tract')
            self.length = tf.placeholder(tf.int64, shape=[None], name='length')

            if self._conf['position_aware']:
                self.position = tf.placeholder(tf.float32, shape=[None, None, 3], name='position')
                #normalize the position
                
            
            if self._conf['normalize_dwi']:
                self.dwi = normalize_dwi_tf(self.dwi)

            if self._conf['position_aware']:
                self.dwi = tf.concat([self.dwi, self.position], axis=2)
            
            self.cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.get_a_cell() for _ in range(self._conf['rnn_depth'])]
            )
            
            self.initial_state = self.cell.zero_state(self._conf['batch_size'], dtype=tf.float32)

            self.outputs, self.last_state = tf.nn.dynamic_rnn(self.cell, 
                                                    self.dwi, 
                                                    initial_state=self.initial_state,
                                                    dtype=tf.float32)
            
            self.predictions = tf.layers.dense(self.outputs, self._conf['output_size'])
            
            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.tract, predictions=self.predictions))
            
            
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.MAX_GRAD_NORM)
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
            
            self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=10)
            
            with tf.variable_scope("performance"):
                self.loss_summary = tf.summary.scalar('loss', self.loss)
                
                self.merged_metric = tf.summary.merge_all()
            
        return self._graph

    def run(self, sess, feed_dict):
        sess.run(self.train_op, feed_dict=feed_dict)

class MultiGRUModel(object):
    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf
        
        self.MAX_GRAD_NORM = 5.0
        self.learning_rate = 1e-3
        
    def get_a_cell(self):
            rnn_cell = tf.nn.rnn_cell.GRUCell(self._conf['num_units'])
            drop = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self._conf['keep_prob'])
            return drop
        
    def build_graph(self):
        with self._graph.as_default():
            self.dwi = tf.placeholder(tf.float32, shape=[None, None, 33])
            self.tract = tf.placeholder(tf.float32, shape=[None, None, 3])
            self.length = tf.placeholder(tf.int64, shape=[None])
            
            if self._conf['normalize_dwi']:
                self.dwi = normalize_dwi_tf(self.dwi)
            
            self.cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.get_a_cell() for _ in range(self._conf['rnn_depth'])]
            )
            
            self.initial_state = self.cell.zero_state(self._conf['batch_size'], dtype=tf.float32)
            
            self.outputs, self.last_state = tf.nn.dynamic_rnn(self.cell, 
                                                    self.dwi, 
                                                    initial_state=self.initial_state,
                                                    dtype=tf.float32)
            
            self.predictions = tf.layers.dense(self.outputs, self._conf['output_size'])
            
            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.tract, predictions=self.predictions))
            
            
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.MAX_GRAD_NORM)
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
            
            
            with tf.variable_scope("performance"):
                self.loss_summary = tf.summary.scalar('loss', self.loss)
                
                self.merged_metric = tf.summary.merge_all()
            
        return self._graph

    def run(self, sess, feed_dict):
        sess.run(self.train_op, feed_dict=feed_dict)


class BiLSTMModel(object):
    
    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf
        
    def build_graph(self):
        
        self.dwi = tf.placeholder(tf.float32, shape=[None, None, 33])
        self.tract = tf.placeholder(tf.float32, shape=[None, None, 3])
        self.length = tf.placeholder(tf.int64, shape=[None])
