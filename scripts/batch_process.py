import os
import sys
sys.path.append('../../')
from deeptrack.utils import FileScanner

_dir = '../../datasets/ISMRM_2015_Tracto_challenge_ground_truth_bundles_TCK_v2/'

files = FileScanner.scan(_dir)
print("get all tck files.")

for trk_name, trk_file in files.items():
    print(trk_file)

    os.system('python3 convert_one.py -t ' + trk_file + ' &')


#os.system('python3 convert_one.py -t ../../datasets/ISMRM_2015_Tracto_challenge_ground_truth_bundles_TRK_v2/CC.trk &')
