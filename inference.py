import tensorflow as tf

from deeptrack.data import DWIDataSet
from deeptrack.utils import Interpolator
from deeptrack.models.seq2seq import MultiLSTMModel
from deeptrack.data import Mask
import os

import nibabel as nib
import numpy as np


class Constrain(object):
    pass

class OutSideMask(Constrain):
    pass

class SharpTurn(Constrain):
    pass


class InferencerBase(object):
    def __init__(self, dwi_data, bval, bvec, mask, model=None, step_size=1, model_dir=None):
        self.dwi_data = dwi_data
        self.bval = bval
        self.bvec = bvec
        self.mask_file = mask
        self.mask = Mask(self.mask_file)
        self.dwi_reader = DWIDataSet(self.dwi_data, self.bval, self.bvec)
        self.interploator = Interpolator(self.dwi_reader, self.dwi_reader.rasmm2vox_affine)

        self._model = model
        # check if the _model is a type of Model base class
        self._graph = self._model.build_graph()
        self._model_dir = model_dir
        
        self.step_size = step_size

    def generate(self, sess, seed, max_length, early_stop=False, threshold=1.0):
        raise NotImplementedError

    def write_to_tck(self, tracts, file_name, affine_to_rasmm=np.eye(4,4)):
        t_gram = nib.streamlines.Tractogram(np.array(tracts), 
                                            affine_to_rasmm=affine_to_rasmm)

        tck_out = nib.streamlines.TckFile(t_gram)

        tck_out.save(file_name)


class InferencerMLP(InferencerBase):
    def __init__(self, dwi_data, bval, bvec, mask, model=None, step_size=1, model_dir=None):
        super().__init__(dwi_data, bval, bvec, mask, model=model, step_size=step_size, model_dir=model_dir)

    def generate(self, sess, seed, max_length, early_stop, threshold=1.0):
        cur_point = seed
        tract = []
        tract.append(cur_point)

        for _ in range(max_length):
            next_dir = self._generate_one(sess, cur_point)

            cur_point = cur_point + next_dir * self.step_size

            tract.append(cur_point[0])

            if not self.mask.in_mask(cur_point[0]):
                return tract

        return tract

    def _generate_one(self, sess, _point):
        _dwi = self.interploator.interpolate(_point)
        _dwi = _dwi.reshape((1,33))

        return sess.run(self._model.pred, feed_dict={self._model.dwi: _dwi})



class InferencerRNN(InferencerBase):

    def __init__(self, dwi_data, bval, bvec, mask, model=None, step_size=1, model_dir=None):
        super().__init__(dwi_data, bval, bvec, mask, model=None, step_size=1, model_dir=None)


    def generate(self, sess, seed, max_length, early_stop=False, threshold=1.0):
        
        cur_point = seed
        tract = []
        tract.append(cur_point)

        new_state = sess.run(self._model.initial_state)

        for _ in range(max_length):

            next_dir, new_state = self._generate_one(sess, cur_point, new_state)

            if early_stop:
                if np.linalg.norm(next_dir) < threshold:
                    return tract    

            cur_point = cur_point + next_dir * self.step_size

            tract.append(cur_point[0][0])

            #check if in the mask
            if not self.mask.in_mask(cur_point[0][0]):
                return tract

        return tract

    def _generate_one(self, sess, _point, _state):
        _dwi = self.interploator.interpolate(_point)

        _dwi = _dwi.reshape((1,1,33))

        input_dict ={
            self._model.dwi: _dwi,
            self._model.initial_state: _state
        }

        return sess.run([self._model.predictions,
                        self._model.last_state],
                        feed_dict=input_dict)


class Inferencer(object):
    def __init__(self, dwi_data, bval, bvec, mask, model=None, step_size=1, model_dir=None):
        self.dwi_data = dwi_data
        self.bval = bval
        self.bvec = bvec
        self.mask_file = mask
        self.mask = Mask(self.mask_file)
        self.dwi_reader = DWIDataSet(self.dwi_data, self.bval, self.bvec)
        self.interploator = Interpolator(self.dwi_reader, self.dwi_reader.rasmm2vox_affine)

        self._model = model
        # check if the _model is a type of Model base class
        self._graph = self._model.build_graph()
        self._model_dir = model_dir
        
        self.step_size = step_size


    def generate(self, sess, seed, max_length, early_stop=False, threshold=1.0):
        
        cur_point = seed
        tract = []
        tract.append(cur_point)

        new_state = sess.run(self._model.initial_state)

        for _ in range(max_length):

            next_dir, new_state = self._generate_one(sess, cur_point, new_state)

            if early_stop:
                if np.linalg.norm(next_dir) < threshold:
                    return tract    

            cur_point = cur_point + next_dir * self.step_size

            tract.append(cur_point[0][0])

            #check if in the mask
            if not self.mask.in_mask(cur_point[0][0]):
                return tract

        return tract

    def _generate_one(self, sess, _point, _state):
        _dwi = self.interploator.interpolate(_point)

        _dwi = _dwi.reshape((1,1,33))

        input_dict ={
            self._model.dwi: _dwi,
            self._model.initial_state: _state
        }

        return sess.run([self._model.predictions,
                        self._model.last_state],
                        feed_dict=input_dict)

    def write_to_tck(self, tracts, file_name, affine_to_rasmm=np.eye(4,4)):
        t_gram = nib.streamlines.Tractogram(np.array(tracts), 
                                            affine_to_rasmm=affine_to_rasmm)

        tck_out = nib.streamlines.TckFile(t_gram)

        tck_out.save(file_name)

class ModelLoader(object):
    def __init__(self, model_dir, model_name, model_checkpoints=None):
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_checkpoints = model_checkpoints
        
    def restore_model(self):
        model_graph_file = os.path.join(self.model_dir, self.model_name + '.meta')

        self.saver = tf.train.import_meta_graph(model_graph_file)
    
    def restore_weights(self, sess):
        _model = tf.train.latest_checkpoint(self.model_dir)

        self.saver.restore(sess, _model)
        
        
