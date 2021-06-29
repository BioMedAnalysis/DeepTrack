import tensorflow as tf

import sys
sys.path.append('../../')
from deeptrack.data import TractTckDataSet
from deeptrack.utils import SimpleDirectionCalculator
from deeptrack.data import DWIDataSet
from deeptrack.utils import Interpolator
from deeptrack.utils import TFRecordsWriterMLP
from deeptrack.utils import TFRecordsReader
from deeptrack.utils import FileScanner
from argparse import ArgumentParser

import numpy as np

dwi_data = '../../datasets/ISMRM_2015_Tracto_challenge_data/Diffusion.nii.gz'
bval = '../../datasets/ISMRM_2015_Tracto_challenge_data/Diffusion.bvals'
bvec = '../../datasets/ISMRM_2015_Tracto_challenge_data/Diffusion.bvecs'
_dir = '../../datasets/ISMRM_2015_Tracto_challenge_ground_truth_bundles_TRK_v2/'

def worker(trk_name,trk_file):
    
    #index = 0

    mlp_tfwriter = TFRecordsWriterMLP(trk_file, dwi_data, bval, bvec, 'mlp-' + trk_name+'.tfrecords')
    
    while True:
        next_line = mlp_tfwriter.next_line()

        if next_line is not None:
            dwi, tract, length = next_line
            
            dir_cal = SimpleDirectionCalculator(tract, length)
            
            temp = zip(dwi[:-1], tract[:-1], dir_cal.estimate()[:-1])
            
            for _dwi, _pos, _dir in temp:
                example = mlp_tfwriter.to_tf_example(np.array(_dwi),np.array(_pos), np.array(_dir))
                mlp_tfwriter.to_tfrecords(example)
                
        else:
            break
            
        # if index == 10:
        #     break
            
        # index+=1

    mlp_tfwriter.close_tfrecords()


def main():

    parser = ArgumentParser()
    parser.add_argument("-t", "--tract", dest="track_file",
                        help="the track file in trk format",
                        required=True)

    args = parser.parse_args() 

    print(args.track_file)
    
    trk_file = args.track_file
    
    print(trk_file.split('/')[-1].split('.')[0])
    print(trk_file)
    
    worker(trk_file.split('/')[-1].split('.')[0], trk_file)
    

if __name__ == '__main__':
    main()
