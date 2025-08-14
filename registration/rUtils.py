import time, os
import numpy as np
from numpy import fft
from numba import vectorize, complex64, float32, int16
import math
from scipy.ndimage import gaussian_filter1d
from mkl_fft import fft2, ifft2

def set_ops(data):
    ops = {
        # parallel settings
        'num_workers': 0,       # 0 to select num_cores, -1 to disable parallelism, N to enforce value
        'num_workers_roi': -1,  # 0 to select number of planes, -1 to disable parallelism, N to enforce value
        # bidirectional phase offset
        'do_bidiphase': False,
        'bidiphase': 0,
        # registration settings
        'keep_movie_raw': True,
        'nimg_init': 500,       # subsampled frames for finding reference image
        'batch_size': 50,      # number of frames per batch
        'maxregshift': 0.05,    # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan' : 1,    # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': False,       # whether to save registered tiffs
        'reg_tif_chan2': False, # whether to save channel 2 registered tiffs
        'subpixel' : 10,        # precision of subpixel registration (1/subpixel steps)
        'smooth_sigma': 2,   # ~1 good for 2P recordings, recommend >5 for 1P recordings
        'smooth_sigma_time' : 0,# gaussian smoothing in time
        'th_badframes': 1.0,    # this parameter determines which frames to exclude when determining cropping - set it smaller to exclude more frames
        'pad_fft': False,
        #non rigid registration settings
        'nonrigid': False,       # whether to use nonrigid registration
        'block_size': [128, 128],# block size to register (** keep this a multiple of 2 **)
        'snr_thresh': 1.2,      # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
        'maxregshiftNR': 5,     # maximum pixel shift allowed for nonrigid, relative to rigid
        # 1P settings
        '1Preg': False,         # whether to perform high-pass filtering and tapering
        'spatial_hp': 50,       # window for spatial high-pass filtering before registration
        'pre_smooth': 2,        # whether to smooth before high-pass filtering before registration
        'spatial_taper': 50,    # how much to ignore on edges (important for vignetted windows, for FFT padding do not set BELOW 3*ops['smooth_sigma'])
        # cell detection settings
        'nframes': data.shape[0],
        'Ly': data.shape[1],
        'Lx': data.shape[2]
      }
    return ops

def one_photon_preprocess(data, ops):
    ''' pre filtering for one-photon data '''
    if ops['pre_smooth'] > 0:
        ops['pre_smooth'] = int(np.ceil(ops['pre_smooth']/2) * 2)
        data = spatial_smooth(data, ops['pre_smooth'])
    else:
        data = data.astype(np.float32)

    #for n in range(data.shape[0]):
    #    data[n,:,:] = laplace(data[n,:,:])
    ops['spatial_hp'] = int(np.ceil(ops['spatial_hp']/2) * 2)
    data = spatial_high_pass(data, ops['spatial_hp'])
    return data

@vectorize([complex64(complex64, complex64)], nopython=True, target = 'parallel')
def apply_dotnorm(Y, cfRefImg):
    eps0 = np.complex64(1e-5)
    x = Y / (eps0 + np.abs(Y))
    x = x*cfRefImg
    return x

def init_offsets(ops):
    """ initialize offsets for all frames """
    yoff = np.zeros((0,),np.float32)
    xoff = np.zeros((0,),np.float32)
    corrXY = np.zeros((0,),np.float32)
    if ops['nonrigid']:
        nb = ops['nblocks'][0] * ops['nblocks'][1]
        yoff1 = np.zeros((0,nb),np.float32)
        xoff1 = np.zeros((0,nb),np.float32)
        corrXY1 = np.zeros((0,nb),np.float32)
        offsets = [yoff,xoff,corrXY,yoff1,xoff1,corrXY1]
    else:
        offsets = [yoff,xoff,corrXY]
    return offsets

def gaussian_fft(sig, Ly, Lx):
    ''' gaussian filter in the fft domain with std sig and size Ly,Lx '''
    x = np.arange(0, Lx)
    y = np.arange(0, Ly)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    hgx = np.exp(-np.square(xx/sig) / 2)
    hgy = np.exp(-np.square(yy/sig) / 2)
    hgg = hgy * hgx
    hgg /= hgg.sum()
    fhg = np.real(fft2(fft.ifftshift(hgg))); # smoothing filter in Fourier domain
    return fhg

def spatial_taper(sig, Ly, Lx):
    ''' spatial taper  on edges with gaussian of std sig '''
    x = np.arange(0, Lx)
    y = np.arange(0, Ly)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    mY = y.max() - 2*sig
    mX = x.max() - 2*sig
    maskY = 1./(1.+np.exp((yy-mY)/sig))
    maskX = 1./(1.+np.exp((xx-mX)/sig))
    maskMul = maskY * maskX
    return maskMul

def spatial_smooth(data,N):
    ''' spatially smooth data using cumsum over axis=1,2 with window N'''
    pad = np.zeros((data.shape[0], int(N/2), data.shape[2]))
    dsmooth = np.concatenate((pad, data, pad), axis=1)
    pad = np.zeros((dsmooth.shape[0], dsmooth.shape[1], int(N/2)))
    dsmooth = np.concatenate((pad, dsmooth, pad), axis=2)
    # in X
    cumsum = np.cumsum(dsmooth, axis=1).astype(np.float32)
    dsmooth = (cumsum[:, N:, :] - cumsum[:, :-N, :]) / float(N)
    # in Y
    cumsum = np.cumsum(dsmooth, axis=2)
    dsmooth = (cumsum[:, :, N:] - cumsum[:, :, :-N]) / float(N)
    return dsmooth

def spatial_high_pass(data, N):
    ''' high pass filters data over axis=1,2 with window N'''
    norm = spatial_smooth(np.ones((1, data.shape[1], data.shape[2])), N).squeeze()
    data -= spatial_smooth(data, N) / norm
    return data
