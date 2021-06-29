import numpy as np
from scipy.interpolate import RegularGridInterpolator
from deeptrack.data import DWIDataSet
from deeptrack.data import TractTckDataSet
import tensorflow as tf
import functools
import glob
import nibabel as nib
from dipy.segment.mask import median_otsu

def normalize_dwi(weights, b0):
    """ Normalize dwi by the first b0.

    Parameters:
    -----------
    weights : ndarray of shape (X, Y, Z, #gradients)
        Diffusion weighted images.
    b0 : ndarray of shape (X, Y, Z)
        B0 image.

    Returns
    -------
    ndarray
        Diffusion weights normalized by the B0.
    """
    b0 = b0[..., None]  # Easier to work if it is a 4D array.

    # Make sure in every voxels weights are lower than ones from the b0.
    # Should not happen, but with the noise we never know!
    nb_erroneous_voxels = np.sum(weights > b0)
    if nb_erroneous_voxels != 0:
        print ("Nb. erroneous voxels: {}".format(nb_erroneous_voxels))
        weights = np.minimum(weights, b0)

    # Normalize dwi using the b0.
    weights_normed = weights / b0
    weights_normed[np.logical_not(np.isfinite(weights_normed))] = 0.

    return weights_normed


def normalize_dwi_tf(dwi):
    b0 = dwi[:,:, 0]

    b0 = b0[...,None]

    dwi_norm = dwi / b0

    dwi_norm_cond = tf.logical_not(tf.is_finite(dwi_norm))

    dwi_norm2 = tf.where(dwi_norm_cond, tf.zeros_like(dwi_norm), dwi_norm)
    
    return dwi_norm2

class Interpolator(object):

    def __init__(self, dwi_reader, affine, method='linear'):
        self.dwi_reader = dwi_reader
        #self.zooms = zooms
        #self.rasmm2vox_affine = np.linalg.inv(affine)
        self.rasmm2vox_affine = affine
        #print(self.rasmm2vox_affine)
        self.method = method

        # align the center of the voxel to (1, 90)
        self._x = np.linspace(0,self.dwi_reader.shape()[0]-1, self.dwi_reader.shape()[0])
        self._y = np.linspace(0,self.dwi_reader.shape()[1]-1, self.dwi_reader.shape()[1])
        self._z = np.linspace(0,self.dwi_reader.shape()[2]-1, self.dwi_reader.shape()[2])

        self.interpolator_collection = []

        self.encoding_dir = self.dwi_reader.shape()[-1]

        self._init()

    def _init(self):
        for each_dir in range(self.encoding_dir):
            self.interpolator_collection.append(
                RegularGridInterpolator((self._x, self._y, self._z), 
                    self.dwi_reader.image_data[:,:,:,each_dir], method=self.method))

        print("interpolater collection init done.")    



    def interpolate(self, _point):

        index = nib.affines.apply_affine(self.rasmm2vox_affine, _point)

        res = [fn(index) for fn in self.interpolator_collection]

        return np.array([x.item() for x in res])
    
    def interpolate_line(self, _line):
        dwi_encoding_line = []
        for _point in _line:
            dwi_encoding_line.append(self.interpolate(_point))
            
        return dwi_encoding_line
            


class Orientation(object):
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z


class DirectionCalculatorBase(object):
    def __init__(self, streamline, length):
        self.streamline = streamline
        self.length = length

    def estimate(self):
        return self._estimate_impl()

class SimpleDirectionCalculator(DirectionCalculatorBase):
    def __init__(self, streamline, length):
        super().__init__(streamline, length)

    def _estimate_impl(self):
        orientations = [self.streamline[each_point +1] - self.streamline[each_point]
                for each_point in range(self.length) if each_point +1 < self.length]

        orientations.append(np.array([0.0,0.0,0.0]))

        return orientations


class OtherDirectionCalculator(DirectionCalculatorBase):
    def __init__(self, streamline, length):
        super().init__(streamline, length)

    def _estimate_impl(self):
        pass


class TFRecordsWriter(object):
    def __init__(self,tract_file, dwi_data, bval, bvec, tfrecord_file):
        self.tract_file = tract_file
        self.dwi_data = dwi_data
        self.bval = bval
        self.bvec = bvec
        
        self.tracts = TractTckDataSet(self.tract_file)
        self.dwi_reader = DWIDataSet(self.dwi_data, bval, bvec)
        
        #self.interploator = Interpolator(self.dwi_reader, self.dwi_reader.get_zooms())
        self.interploator = Interpolator(self.dwi_reader, self.dwi_reader.rasmm2vox_affine)
        
        self.tfrecord_file = tfrecord_file
        self.tfrecords_writer = tf.python_io.TFRecordWriter(self.tfrecord_file)
        
        
    def next_line(self):
        _line, length = self.tracts.next_line()
        
        if _line is not None:
            dwi_encoding = self.interploator.interpolate_line(_line)
        
            return dwi_encoding, _line, length
        
        return None
    
    def to_tf_example(self, dwi_encoding, _position, _line, _length):
        example = tf.train.SequenceExample()
        
        #feature = {"dwi": tf.train.Feature()
                   #"tract": tf.train.Feature()}
        
        dwi = example.feature_lists.feature_list['dwi']
        position = example.feature_lists.feature_list['position']
        tract = example.feature_lists.feature_list['tract']
        length = example.context.feature['length']
        
        for each in dwi_encoding:
            dwi.feature.add().float_list.value.extend(each)

        for each in _position:
            position.feature.add().float_list.value.extend(each)

        for each in _line:
            tract.feature.add().float_list.value.extend(each)
            
        length = length.int64_list.value.append(_length)
            
        return example

        
    def to_tfrecords(self, example):
        self.tfrecords_writer.write(example.SerializeToString())
        
    
    def close_tfrecords(self):
        self.tfrecords_writer.close()
    
    
class TFRecordsWriterMLP(TFRecordsWriter):
    def __init__(self, tract_file, dwi_data, bval, bvec, tfrecords_file):
        super().__init__(tract_file, dwi_data, bval, bvec, tfrecords_file)

    def to_tf_example(self, dwi_encoding, position, _dir):
        feature = {'dwi': tf.train.Feature(float_list=tf.train.FloatList(value=dwi_encoding)),
                    'position': tf.train.Feature(float_list=tf.train.FloatList(value=position)),
                    'direction': tf.train.Feature(float_list=tf.train.FloatList(value=_dir))}

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        return example


    
class TFRecordsReader(object):
    def __init__(self, tfrecord_file, batch_size=128, max_length=100):
        self.tfrecord_file = tfrecord_file
        self.reader_iterator = tf.python_io.tf_record_iterator(path=self.tfrecord_file)
        
        self.batch_size = batch_size
        self.max_length = max_length
        self.dataset = tf.data.TFRecordDataset(self.tfrecord_file)
        
        self._init()
        
    def next_example(self):
        example = tf.train.SequenceExample()
        
        try:
            example.ParseFromString(next(self.reader_iterator))
            dwi = example.feature_lists.feature_list.get('dwi').feature

            tract = example.feature_lists.feature_list.get('tract').feature
            
            dwi_list = []
            tract_list = []
            
            for each in dwi:
                dwi_list.append(list(each.float_list.value))
                
            for each in tract:
                tract_list.append(list(each.float_list.value))
                
            return np.array(dwi_list), np.array(tract_list)    
            
        except StopIteration:
            return None
        
        
    def next_batch_deprecated(self, batch_size):
        dwi_batch = []
        tract_batch = []
        
        for each in range(batch_size):
            dwi, tract = self.next_example()
            
            if dwi is not None and tract is not None:
                dwi_batch.append(dwi)
                tract_batch.append(tract)
                
            else:
                return np.array(dwi_batch), np.array(tract_batch)
            
        return np.array(dwi_batch), np.array(tract_batch)
    
    
    def _parser(self, _exmaple, max_length):

        features = tf.parse_single_sequence_example(_exmaple,
                                                sequence_features={
                                                    'dwi': tf.VarLenFeature(tf.float32),
                                                    'position': tf.VarLenFeature(tf.float32),
                                                    'tract':tf.VarLenFeature(tf.float32)
                                                },
                                                context_features={
                                                    'length': tf.FixedLenFeature([], dtype=tf.int64)})    
        
        dwi = tf.sparse.to_dense(features[1]['dwi'])
        position = tf.sparse.to_dense(features[1]['position'])
        tract = tf.sparse.to_dense(features[1]['tract'])
        length = features[0]['length']
        
        #padding
        dwi_padded = tf.pad(dwi, [[0,max_length],[0, 0]])
        position_padded = tf.pad(position, [[0, max_length], [0, 0]])
        tract_padded = tf.pad(tract, [[0,max_length],[0, 0]])
        
        #truncate
        dwi_trun = dwi_padded[:max_length, :]
        position_trun = position_padded[:max_length, :]
        tract_trun = tract_padded[:max_length, :]
        
        #return features['dwi'], features['tract']
        return dwi_trun,position_trun, tract_trun, length


    def _init(self):
        
        self.decoder = functools.partial(self._parser, max_length=self.max_length)
        
        self.dataset = self.dataset.map(self.decoder)
        
        self.dataset = self.dataset.repeat().shuffle(buffer_size=10000).batch(self.batch_size)
        
        self.iterator = self.dataset.make_one_shot_iterator()
        
    def next_batch(self):
        return self.iterator.get_next()
    
class TFRecordsReaderMLP(TFRecordsReader):
    def __init__(self, tfrecord_file, batch_size=128):
        super().__init__(tfrecord_file, batch_size)

    def _parser(self, _example):
        example = tf.parse_single_example(_example,
                            {'dwi': tf.FixedLenFeature([33], dtype=tf.float32),
                            'position': tf.FixedLenFeature([3], dtype=tf.float32),
                            'direction': tf.FixedLenFeature([3], dtype=tf.float32)})

        return example

    def _init(self):
        self.dataset = self.dataset.map(self._parser)

        self.dataset = self.dataset.repeat().shuffle(buffer_size=10000).batch(self.batch_size)

        self.iterator = self.dataset.make_one_shot_iterator()

    


class FileScanner(object):
    @staticmethod
    def scan(_dir, file_type='tck'):
        filenames = glob.glob(_dir + '*.' + str(file_type))
        
        return {filename.split('/')[-1].split('.')[0]: filename for filename in filenames}


class DatasetAggregator(object):
    def __init__(self, dataset_dir, data_type='tfrecords', flag='seq', batch_size=128, max_length=100, exclude_list=None, include_list=None):
        self._dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.flag = flag
        self.exclude_list = exclude_list
        self.include_list = include_list
        self.files = FileScanner.scan(self._dataset_dir, data_type)
        
        self.files = self.filter_files()
        
        
        if self.flag == 'seq':
            self.iterator_list = [TFRecordsReader(_filename, 
                                                   self.batch_size, 
                                                   max_length).next_batch() for _id, _filename in self.files.items()]
        elif self.flag == 'mlp':
            self.iterator_list = [TFRecordsReaderMLP(_filename,
                                                    self.batch_size).next_batch() for _id, _filename in self.files.items()]
        
        self.total_files = len(self.files.keys())
        
        self.iter_index = 0
        
    def next_batch(self):
        self.iter_index += 1
        
        if self.iter_index-1 < self.total_files:
            return self.iterator_list[self.iter_index-1]
        
        else:
            self.iter_index = 0
            return self.iterator_list[self.iter_index]
        
        
    def filter_files(self):
        if self.include_list is not None:
            return {k:v for k,v in self.files.items() if k in self.include_list}
        
        if self.exclude_list is not None:
            return {k:v for k,v in self.files.items() if k not in self.exclude_list}
        
            
        return self.files


class MaskGenerator(object):
    def __init__(self, dwi_file):
        self.dwi_file = dwi_file

        self.dwi_data = nib.load(self.dwi_file)

        self._data = self.dwi_data.get_data()

        self.shape = self._data.shape
        self.affine = self.dwi_data.affine

    def generate_mask(self):
        self.b0_mask, self.mask = median_otsu(data, 2, 1)

    def save_mask(self, filename='mask.nii.gz'):
        nib.save(self.Nifti1Image(self.mask.astype(np.float32), self.affine), filename)                

    def save_b0_mask(self, filename='b0_mask.nii.gz'):
        nib.save(self.Nifti1Image(self.b0_mask.astype(np.float32), self.affine), filename)





