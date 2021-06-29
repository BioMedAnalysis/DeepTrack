from deeptrack.models.seq2seq import MultiRNNModel
import tensorflow as tf
import numpy as np
from deeptrack.utils import DatasetAggregator


# build the graph
conf = {
    'num_units': 128,
    'batch_size': 128,
    'output_size': 3,
    'rnn_depth': 2,
    'keep_prob': 0.5,
    'normalize_dwi': True
}

NUM_STEPS = 200000

rnn_model = MultiRNNModel(conf)
_graph = rnn_model.build_graph()


# load data aggrator
tf_dir = "/home/szho42/workspace_dtrack/workspace/deeptrack/tfrecords/v4/"
batch_size = 128
max_length = 100

tb_log = "/home/szho42/workspace_dtrack/workspace/tb_log/"
model_name = type(rnn_model).__name__

train_summary_writer = tf.summary.FileWriter(tb_log + model_name + "/train")
train_summary_writer.add_graph(_graph)
val_summary_writer = tf.summary.FileWriter(tb_log + model_name + "/val")


from deeptrack.utils import FileScanner

tf_dir = "/home/szho42/workspace_dtrack/workspace/deeptrack/tfrecords/v4/"
_files = FileScanner.scan(tf_dir, file_type='tfrecords')

all_data = _files.keys()
val_include_list = ['CA','CP', 'SCP_right', 'POPT_right']

train_include_list = all_data - val_include_list

# training

with tf.Session(graph=_graph) as sess:
    
    #tfreader = TFRecordsReader(tffile, batch_size, max_length)
    #training data aggregator
    
    train_data_aggregator = DatasetAggregator(tf_dir, 
                                              'tfrecords',
                                              batch_size, 
                                              max_length,
                                             include_list=train_include_list)

    # validation data aggregator
    val_data_aggregator = DatasetAggregator(tf_dir, 
                                            'tfrecords',
                                           batch_size,
                                           max_length,
                                           include_list=val_include_list)


    
    
    sess.run(tf.global_variables_initializer())
    
    # model meta and checkpoints saver
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=10)
    saver.export_meta_graph(tb_log + model_name + "/" + model_name + ".meta")
   
    # train 
    for step in range(NUM_STEPS):
        
        dwi, tract, length = sess.run(train_data_aggregator.next_batch())
        
        feed_dict = {rnn_model.dwi: dwi,
            rnn_model.tract: tract}
        
        rnn_model.run(sess, feed_dict)
        
        if step % 100 == 0:
            # save summary and checkpoints
            _loss, _loss_metric = sess.run([rnn_model.loss, rnn_model.merged_metric], feed_dict=feed_dict)
            
            train_summary_writer.add_summary(_loss_metric, step)

            val_dwi, val_tract, val_length = sess.run(val_data_aggregator.next_batch())
            val_feed_dict = {rnn_model.dwi: val_dwi,
                rnn_model.tract: val_tract}
            
            val_loss, val_loss_metric = sess.run([rnn_model.loss, rnn_model.merged_metric], feed_dict=val_feed_dict)

            val_summary_writer.add_summary(val_loss_metric, step)
            
            
            if step %1000 == 0:
                saver.save(sess, tb_log+model_name + "/" + model_name, global_step=step)
