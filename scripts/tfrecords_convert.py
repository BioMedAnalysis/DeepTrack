import tensorflow as tf
#tf.enable_eager_execution()

import sys
sys.path.append('/home/szho42/workspace_dtrack/workspace')
from deeptrack.data import TractTckDataSet
from deeptrack.utils import SimpleDirectionCalculator
from deeptrack.data import DWIDataSet
from deeptrack.utils import Interpolator
from deeptrack.utils import TFRecordsWriter
from deeptrack.utils import TFRecordsReader
from deeptrack.utils import FileScanner

import threading

#trk_file = './datasets/ISMRM_2015_Tracto_challenge_ground_truth_bundles_TRK_v2/CC.trk'
dwi_data = './datasets/ISMRM_2015_Tracto_challenge_data/Diffusion.nii.gz'
bval = './datasets/ISMRM_2015_Tracto_challenge_data/Diffusion.bvals'
bvec = './datasets/ISMRM_2015_Tracto_challenge_data/Diffusion.bvecs'
_dir = './datasets/ISMRM_2015_Tracto_challenge_ground_truth_bundles_TCK_v2/'

def worker(trk_name,trk_file):
    #index = 0
    
    tfwriter = TFRecordsWriter(trk_file, dwi_data, bval, bvec, 
                                 trk_name+'.tfrecords')
    
    while True:
        next_line = tfwriter.next_line()

        if next_line is not None:
            dwi, tract, length = next_line
            
            dir_cal = SimpleDirectionCalculator(tract, length)

            example = tfwriter.to_tf_example(dwi, dir_cal.estimate(), length)

            tfwriter.to_tfrecords(example)
            
        else:
            break
            
        #if index == 10:
            #break
            
        #index+=1

    tfwriter.close_tfrecords()

def main():
    files = FileScanner.scan(_dir)
    print("get all trk files.")
    print(files)
    
    for trk_name,trk_file in files.items():
        t = threading.Thread(target=worker, args = (trk_name,trk_file))
        print("thread: {} starts...".format(trk_name))
        #t.daemon = True
        t.start()

if __name__ == '__main__':
    main()
