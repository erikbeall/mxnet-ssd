import mxnet as mx
import numpy as np
import cv2
from tools.image_processing import resize, transform
from tools.rand_sampler import RandSampler
import random
from scipy import misc
from scipy import ndimage
from scipy.stats import chi2,norm

class DetIter(mx.io.DataIter):
    """
    Detection Iterator, which will feed data and label to network
    Optional data augmentation is performed when providing batch

    Parameters:
    ----------
    imdb : Imdb
        image database
    batch_size : int
        batch size
    data_shape : int or (int, int)
        image shape to be resized
    mean_pixels : float or float list
        [R, G, B], mean pixel values
    rand_samplers : list
        random cropping sampler list, if not specified, will
        use original image only
    rand_mirror : bool
        whether to randomly mirror input images, default False
    shuffle : bool
        whether to shuffle initial image list, default False
    rand_seed : int or None
        whether to use fixed random seed, default None
    max_crop_trial : bool
        if random crop is enabled, defines the maximum trial time
        if trial exceed this number, will give up cropping
    is_train : bool
        whether in training phase, default True, if False, labels might
        be ignored
    """
    def __init__(self, imdb, batch_size, data_shape, \
                 mean_pixels=[128, 128, 128], rand_samplers=[], \
                 rand_mirror=False, shuffle=False, rand_seed=None, \
                 is_train=True, max_crop_trial=50):
        super(DetIter, self).__init__()

        self._imdb = imdb
        self.batch_size = batch_size
        if isinstance(data_shape, int):
            data_shape = (data_shape, data_shape)
        self._data_shape = data_shape
        self._mean_pixels = mean_pixels
        if not rand_samplers:
            self._rand_samplers = []
        else:
            if not isinstance(rand_samplers, list):
                rand_samplers = [rand_samplers]
            assert isinstance(rand_samplers[0], RandSampler), "Invalid rand sampler"
            self._rand_samplers = rand_samplers
        self.is_train = is_train
        self._rand_mirror = rand_mirror
        self._shuffle = shuffle
        if rand_seed:
            np.random.seed(rand_seed) # fix random seed
        self._max_crop_trial = max_crop_trial

        # create mask to clamp edges, multiply by mask that goes to zero radially
        mask=np.ones(self._data_shape,dtype=np.float)
        mask[0:int(0.05*self._data_shape[0]),:]=0;
        mask[int(0.95*self._data_shape[0]):,:]=0;
        mask[:,0:int(0.05*self._data_shape[1])]=0;
        mask[:,int(0.95*self._data_shape[1]):]=0;
        self._maskf=ndimage.gaussian_filter(mask, sigma=15.0, mode='reflect')

        # create warp function interpolator
        xpos=np.arange(0,self._data_shape[0],1)
        ypos=np.arange(0,self._data_shape[1],1)
        xx, yy = np.meshgrid(xpos, ypos)
        self._xx=xx
        self._yy=yy

        self._current = 0
        self._size = imdb.num_images
        self._index = np.arange(self._size)

        self._data = None
        self._label = None
        self._get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in self._data.items()]

    @property
    def provide_label(self):
        if self.is_train:
            return [(k, v.shape) for k, v in self._label.items()]
        else:
            return []

    def reset(self):
        self._current = 0
        if self._shuffle:
            np.random.shuffle(self._index)

    def iter_next(self):
        return self._current < self._size

    def next(self):
        if self.iter_next():
            self._get_batch()
            data_batch = mx.io.DataBatch(data=self._data.values(),
                                   label=self._label.values(),
                                   pad=self.getpad(), index=self.getindex())
            self._current += self.batch_size
            return data_batch
        else:
            raise StopIteration

    def getindex(self):
        return self._current / self.batch_size

    def getpad(self):
        pad = self._current + self.batch_size - self._size
        return 0 if pad < 0 else pad

    def _get_batch(self):
        """
        Load data/label from dataset
        """
        batch_data = []
        batch_label = []
        for i in range(self.batch_size):
            if (self._current + i) >= self._size:
                if not self.is_train:
                    continue
                # use padding from middle in each epoch
                idx = (self._current + i + self._size / 2) % self._size
                index = self._index[idx]
            else:
                index = self._index[self._current + i]
            # index = self.debug_index
            im_path = self._imdb.image_path_from_index(index)
            img = cv2.imread(im_path)
            gt = self._imdb.label_from_index(index).copy() if self.is_train else None
            data, label = self._data_augmentation(img, gt)
            batch_data.append(data)
            if self.is_train:
                batch_label.append(label)
        # pad data if not fully occupied
        for i in range(self.batch_size - len(batch_data)):
            assert len(batch_data) > 0
            batch_data.append(batch_data[0] * 0)
        self._data = {'data': mx.nd.array(np.array(batch_data))}
        if self.is_train:
            self._label = {'label': mx.nd.array(np.array(batch_label))}
        else:
            self._label = {'label': None}

    def _data_augmentation(self, data, label):
        """
        perform data augmentations: crop, mirror, resize, sub mean, swap channels...
        """
        # _rand_samplers is set to config.TRAIN.RAND_SAMPLERS list when training
        if self.is_train and self._rand_samplers:
            rand_crops = []
            for rs in self._rand_samplers:
                rand_crops += rs.sample(label)
            num_rand_crops = len(rand_crops)
            # randomly pick up one as input data
            if num_rand_crops > 0:
                index = int(np.random.uniform(0, 1) * num_rand_crops)
                width = data.shape[1]
                height = data.shape[0]
                crop = rand_crops[index][0]
                xmin = int(crop[0] * width)
                ymin = int(crop[1] * height)
                xmax = int(crop[2] * width)
                ymax = int(crop[3] * height)
                if xmin >= 0 and ymin >= 0 and xmax <= width and ymax <= height:
                    data = data[ymin:ymax, xmin:xmax, :]
                else:
                    # padding mode
                    new_width = xmax - xmin
                    new_height = ymax - ymin
                    offset_x = 0 - xmin
                    offset_y = 0 - ymin
                    data_bak = data
                    data = np.full((new_height, new_width, 3), 128.)
                    # EBB could fill with random data, although will increase time taken
                    data[offset_y:offset_y+height, offset_x:offset_x + width, :] = data_bak
                label = rand_crops[index][1]

        if self.is_train and self._rand_mirror:
            if np.random.uniform(0, 1) > 0.5:
                data = cv2.flip(data, 1)
                valid_mask = np.where(label[:, 0] > -1)[0]
                tmp = 1.0 - label[valid_mask, 1]
                label[valid_mask, 1] = 1.0 - label[valid_mask, 3]
                label[valid_mask, 3] = tmp

        if self.is_train:
            # EBB get rid of LANCZOS4 
            # not a big difference with CUBIC and takes more time. 
            # Worse, it introduces some artifacts with strong white/black (ringing) which I expect with geese at least.
            interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, \
                              cv2.INTER_NEAREST]
        else:
            interp_methods = [cv2.INTER_LINEAR]
        interp_method = interp_methods[int(np.random.uniform(0, 1) * len(interp_methods))]
        data = resize(data, self._data_shape, interp_method)

        # EBB insert augmentations
        if self.is_train:
            # displacement field MxNx2 (x,y displacements in pixel units) where MxN defines the output matrix size
            # 2D slicing
            warpfield=np.zeros((data.shape[0],data.shape[1],2))
            warpfield[:,:,0]=misc.imresize(norm.rvs(loc=0,scale=2.5,size=(data.shape[0]/32,data.shape[1]/32)),(data.shape[0],data.shape[1]),mode='F')*self._maskf
            warpfield[:,:,1]=misc.imresize(norm.rvs(loc=0,scale=2.5,size=(data.shape[0]/32,data.shape[1]/32)),(data.shape[0],data.shape[1]),mode='F')*self._maskf
            # adjust the inputmesh grid by the warpfield
            xxw=self._xx+np.round(warpfield[:,:,0])
            yyw=self._yy+np.round(warpfield[:,:,1])
            xxw=xxw.astype(int)
            yyw=yyw.astype(int)
            xxw[xxw>np.max(self._xx)]=np.max(self._xx)
            yyw[yyw>np.max(self._yy)]=np.max(self._yy)
            xxw[xxw<np.min(self._xx)]=np.min(self._xx)
            yyw[yyw<np.min(self._yy)]=np.min(self._yy)
            dataw=np.copy(data[yyw,xxw])
            # speckle noise - add these in double format, rescale and fit to uint8 range - note, one for each channel
            speckle=chi2.rvs(abs(random.normalvariate(15,5)),size=data.shape);
            # speckle alpha parameter - max it out at 0.3
            alpha_speckle=min(abs(random.normalvariate(0.15,0.1)),0.25);
            # fog - blur a speckle map and add it - note, same one across all channels (assume its in the optical path)
            fog=ndimage.uniform_filter(chi2.rvs(abs(random.normalvariate(15,7)),size=data.shape[0:2]), 15, mode='reflect');
            fog=(fog-np.min(fog))*256/(np.max(fog)-np.min(fog));
            # max fog alpha parameter at 0.35
            alpha_fog=min(abs(random.normalvariate(0.1,0.1)),0.25);
            # blur
            datawb=np.copy(dataw);
            # max out blur at 3
            blurfact=max(min(abs(random.normalvariate(1.25,1.5)),1.5),0.5)
            datawb[:,:,0]=ndimage.gaussian_filter(dataw[:,:,0], sigma=blurfact, mode='reflect')
            datawb[:,:,1]=ndimage.gaussian_filter(dataw[:,:,1], sigma=blurfact, mode='reflect')
            datawb[:,:,2]=ndimage.gaussian_filter(dataw[:,:,2], sigma=blurfact, mode='reflect')
            data=np.copy(datawb)
            for chan in range(0,3,1):
                data[:,:,chan]=np.uint8(((1-alpha_fog-alpha_speckle)*np.double(datawb[:,:,chan]))+alpha_fog*fog+speckle[:,:,chan]*alpha_speckle);

        # original functions: transform data (mean pixel removal) and return the resulting label and image
        data = transform(data, self._mean_pixels)
        return data, label
