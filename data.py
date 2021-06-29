import nibabel as nib
import typing
import itertools
import numpy as np

class DWIDataSet(object):
    def __init__(self, dwi_data: str, bval: str, bvec: str) -> None:
        self.dwi_data = dwi_data
        self.bval = bval
        self.bvec = bvec

        self.image_data, self.header = self._load_image()
        
        self.vox2rasmm_affine = self.img_obj.affine
        self.rasmm2vox_affine = np.linalg.inv(self.vox2rasmm_affine)

    def _load_image(self):
        self.img_obj = nib.load(self.dwi_data)

        return self.img_obj.get_data(), self.img_obj.header

    def shape(self):
        return self.image_data.shape

    def get_voxel(self,x,y,z):
        return self.image_data[x,y,z,:]

    def interpolate(self, point):
        _x,_y,_z = point
        pass

    def get_zooms(self):
        return self.header.get_zooms()



class TractTrkDataSet(object):
    
    def __init__(self, trk_file:str) -> None:

        self.trk_file = trk_file

        if trk_file.split(".")[-1] != 'trk':
            raise ValueError("Only trk format is supported.")

        self.streamlines, self.header = nib.trackvis.read(self.trk_file)

        self._generator()


    def line_count(self):
        return len(self.streamlines)

    def point_count(self):
        return None

    def _generator(self):
        self._streamlines_iter = itertools.chain(self.streamlines)

    def next_line(self):
        try:
            temp = next(self._streamlines_iter)
        except StopIteration:
            print("Run out of Streamlines")
            return None

        return temp[0], len(temp[0])


class TractTckDataSet(object):
    
    def __init__(self, tck_file:str) -> None:

        self.tck_file = tck_file

        if tck_file.split(".")[-1] != 'tck':
            raise ValueError("Only tck format is supported.")

        self.tck_obj = nib.streamlines.tck.TckFile(self.tck_file)
        
        self.streamlines = self._load().streamlines

        self._iter = self._generator()

    def _load(self):
        with open(self.tck_file, 'rb') as f:
            return self.tck_obj.load(f)

    def line_count(self):
        return len(self.streamlines)

    def point_count(self):
        return None
    
    def _generator(self):
        for each in range(self.line_count()):
            yield self.streamlines[each]

    def next_line(self):
        try: 
            temp = next(self._iter)
        except StopIteration:
            return None, None
        
        return temp, temp.shape[0]

class TractTckDataWriter(object):
    def __init__(self, tck_file:str) -> None:
        pass

    def write(self):
        pass
        

class Mask(object):
    def __init__(self, mask_file):
        assert isinstance(mask_file, str)

        self.mask = nib.load(mask_file)
        self.data = self.mask.get_data()
        self.shape = self.mask.shape
        self.vox2rasmm_affine = self.mask.affine
        self.rasmm2vox_affine = np.linalg.inv(self.vox2rasmm_affine)

    def in_mask(self, _point):
        index = nib.affines.apply_affine(self.rasmm2vox_affine, _point)

        assert int(index[0]) < self.shape[0]
        assert int(index[1]) < self.shape[1]
        assert int(index[2]) < self.shape[2]

        return self.data[int(index[0]), int(index[1]), int(index[2])]
    

