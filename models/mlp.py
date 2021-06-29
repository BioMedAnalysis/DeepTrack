import tensorflow as tf
import sonnet as snt
from deeptrack.models.utils import normalize_dwi_tf_1d

class SimpleMLP(object):
    
    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf
        
    def build_graph(self):
        
        with self._graph.as_default():
            self.dwi = tf.placeholder(tf.float32, shape=[None, 33], name='dwi') # (batch_size, dwi)
            self.dir = tf.placeholder(tf.float32, shape=[None, 3], name='direction') # (batch_size, dir)

            layer1 = snt.Linear(128)
            layer2 = snt.Linear(64)
            layer3 = snt.Linear(3)

            net = tf.nn.relu(layer1(self.dwi))
            net = tf.nn.relu(layer2(net))
            self.pred = layer3(net)

            #self.pred = layer3(layer2(layer1(self.dwi)))

            self.loss = tf.losses.mean_squared_error(labels=self.dir,
                                                    predictions=self.pred)

            self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)

            self.init_op = tf.global_variables_initializer()

            self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=10)
                
            with tf.variable_scope("performance"):
                self.loss_summary = tf.summary.scalar('loss', self.loss)
                    
                self.merged_metric = tf.summary.merge_all()


        return self._graph

    def train(self, sess, feed_dict):
        sess.run(self.optimizer, feed_dict=feed_dict)

    def validation(self, sess, feed_dict):
        return sess.run([self.loss, self.merged_metric], feed_dict=feed_dict)


