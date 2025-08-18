import time, os
import numpy as np
from scipy.fftpack import next_fast_len
from numpy import fft
from numba import vectorize, complex64, float32, int16
import math
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
# from suite2p.io import tiff
from mkl_fft import fft2, ifft2
from . import reference, bidiphase, nonrigid, rUtils, rigid


# HAS_GPU=True
# try:
#     import cupy as cp
#     from cupyx.scipy.fftpack import fftn, ifftn, get_fft_plan
#     HAS_GPU=True
# except ImportError:
#     HAS_GPU=False
#     print('GPU')

def prepare_refAndMasks(refImg, ops):
    """ prepares refAndMasks for phasecorr using refImg

    Parameters
    ----------
    refImg : int16
        reference image

    ops : dictionary
        requires 'smooth_sigma'
        (if ```ops['1Preg']```, need 'spatial_taper', 'spatial_hp', 'pre_smooth')

    Returns
    -------
    refAndMasks : list
        maskMul, maskOffset, cfRefImg (see register.prepare_masks for details)

    """
    maskMul, maskOffset, cfRefImg = rigid.phasecorr_reference(refImg, ops)
    if 'nonrigid' in ops and ops['nonrigid']:
        maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.phasecorr_reference(refImg, ops)
        refAndMasks = [maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR]
    else:
        refAndMasks = [maskMul, maskOffset, cfRefImg]
    return refAndMasks

def compute_motion_and_shift(data, refAndMasks, ops):
    """ register data matrix to reference image and shift

    need to run ```refAndMasks = register.prepare_refAndMasks(ops)``` to get fft'ed masks;
    if ```ops['nonrigid']``` need to run ```ops = nonrigid.make_blocks(ops)```

    Parameters
    ----------
    data : int16
        array that's frames x Ly x Lx
    refAndMasks : list
        maskMul, maskOffset and cfRefImg (from prepare_refAndMasks)
    ops : dictionary
        requires 'nonrigid', 'bidiphase', '1Preg'

    Returns
    -------
    data : int16
        registered frames x Ly x Lx
    ymax : int
        shifts in y from cfRefImg to data for each frame
    xmax : int
        shifts in x from cfRefImg to data for each frame
    cmax : float
        maximum of phase correlation for each frame
    yxnr : list
        ymax, xmax and cmax from the non-rigid registration

    """

    if ops['bidiphase']!=0 and not ops['bidi_corrected']:
        bidiphase.shift(data, ops['bidiphase'])
    nr=False
    yxnr = []
    if ops['nonrigid'] and len(refAndMasks)>3:
        nb = ops['nblocks'][0] * ops['nblocks'][1]
        nr=True

    # rigid registration
    if ops['smooth_sigma_time'] > 0: # temporal smoothing:
        data_smooth = gaussian_filter1d(data.copy(), sigma=ops['smooth_sigma_time'], axis=0)
        ymax, xmax, cmax = rigid.phasecorr(data_smooth, refAndMasks[:3], ops)
    else:
        ymax, xmax, cmax = rigid.phasecorr(data, refAndMasks[:3], ops)
    rigid.shift_data(data, ymax, xmax)

    # non-rigid registration
    if nr:
        if ops['smooth_sigma_time'] > 0: # temporal smoothing:
            data_smooth = gaussian_filter1d(data.copy(), sigma=ops['smooth_sigma_time'], axis=0)
            ymax1, xmax1, cmax1, _ = nonrigid.phasecorr(data_smooth, refAndMasks[3:], ops)
        else:
            ymax1, xmax1, cmax1, _ = nonrigid.phasecorr(data, refAndMasks[3:], ops)
        yxnr = [ymax1,xmax1,cmax1]
        data = nonrigid.transform_data(data, ops, ymax1, xmax1)
    return data, ymax, xmax, cmax, yxnr

def compute_crop(ops):
    ''' determines ops['badframes'] (using ops['th_badframes'])
        and excludes these ops['badframes'] when computing valid ranges
        from registration in y and x
    '''
    dx = ops['xoff'] - medfilt(ops['xoff'], 101)
    dy = ops['yoff'] - medfilt(ops['yoff'], 101)
    # offset in x and y (normed by mean offset)
    dxy = (dx**2 + dy**2)**.5
    dxy /= dxy.mean()
    # phase-corr of each frame with reference (normed by median phase-corr)
    cXY = ops['corrXY'] / medfilt(ops['corrXY'], 101)
    # exclude frames which have a large deviation and/or low correlation
    px = dxy / np.maximum(0, cXY)
    ops['badframes'] = np.logical_or(px > ops['th_badframes'] * 100, ops['badframes'])
    ops['badframes'] = np.logical_or(abs(ops['xoff']) > (ops['maxregshift'] * ops['Lx'] * 0.95), ops['badframes'])
    ops['badframes'] = np.logical_or(abs(ops['yoff']) > (ops['maxregshift'] * ops['Ly'] * 0.95), ops['badframes'])
    ymin = np.maximum(0, np.ceil(np.amax(ops['yoff'][np.logical_not(ops['badframes'])])))
    ymax = ops['Ly'] + np.minimum(0, np.floor(np.amin(ops['yoff'])))
    xmin = np.maximum(0, np.ceil(np.amax(ops['xoff'][np.logical_not(ops['badframes'])])))
    xmax = ops['Lx'] + np.minimum(0, np.floor(np.amin(ops['xoff'])))
    ops['yrange'] = [int(ymin), int(ymax)]
    ops['xrange'] = [int(xmin), int(xmax)]
    return ops

def apply_shifts(data, ops, ymax, xmax, ymax1, xmax1):
    """ apply rigid and nonrigid shifts to data (for chan that's not 'align_by_chan')

    Parameters
    ----------
    data : int16


    ops : dictionary

    refImg : int16
        reference image

    reg_file_align : string
        file to (read if raw_file_align empty, and) write registered binary to

    raw_file_align : string
        file to read raw binary from (if not empty)

    Returns
    -------
    ops : dictionary
        sets 'meanImg' or 'meanImg_chan2'
        maskMul, maskOffset, cfRefImg (see register.prepare_masks for details)

    offsets : list
        [ymax, xmax, cmax, yxnr] <- shifts and correlations


    """
    if ops['bidiphase']!=0  and not ops['bidi_corrected']:
        bidiphase.shift(data, ops['bidiphase'])
    rigid.shift_data(data, ymax, xmax)
    if ops['nonrigid']==True:
        data = nonrigid.transform_data(data, ops, ymax1, xmax1)
    return data

def register_npy(Z, ops=None, refImg=None):
    #Use default options; i.e. just basic rigid registration
    if ops is None:
        ops = rUtils.set_ops(Z)

    # Get a reference image
    if refImg is None:
        refImg = reference.compute_reference_image(Z,ops)
#     import pdb; pdb.set_trace()
    #Non-rigid alignment?
    if ops['nonrigid']:
        ops = rUtils.make_blocks(ops)

    # Preallocate arrays
    nframes, Ly, Lx = Z.shape
    meanImg = np.zeros((Ly, Lx)) 
    yoff = np.zeros((0,),np.float32)
    xoff = np.zeros((0,),np.float32)
    corrXY = np.zeros((0,),np.float32)

    if ops['nonrigid']:
        yoff1 = np.zeros((0,nb),np.float32)
        xoff1 = np.zeros((0,nb),np.float32)
        corrXY1 = np.zeros((0,nb),np.float32)

    refAndMasks = prepare_refAndMasks(refImg, ops) # prepare masks for rigid registration
    k = 0
    nfr = 0
    Zreg = np.zeros((nframes, Ly, Lx,), 'uint16')
    while True:
        irange = np.arange(nfr, min(nfr+ops['batch_size'],nframes))
        data = Z[irange, :,:]
        if data.size==0:
            break
        data = np.reshape(data, (-1, Ly, Lx))
        dwrite, ymax, xmax, cmax, yxnr = compute_motion_and_shift(data, refAndMasks, ops)
        
        dwrite = dwrite.astype('uint16') # need to hold on to this
        meanImg += dwrite.sum(axis=0)
        yoff = np.hstack((yoff, ymax))
        xoff = np.hstack((xoff, xmax))
        corrXY = np.hstack((corrXY, cmax))
        if ops['nonrigid']:
            yoff1 = np.vstack((yoff1, yxnr[0]))
            xoff1 = np.vstack((xoff1, yxnr[1]))
            corrXY1 = np.vstack((corrXY1, yxnr[2]))
        nfr += dwrite.shape[0]
        Zreg[irange] = dwrite

    # compute some potentially useful info
    ops['th_badframes'] = 100
    dx = xoff - medfilt(xoff, 101)
    dy = yoff - medfilt(yoff, 101)
    dxy = (dx**2 + dy**2)**.5
    cXY = corrXY / medfilt(corrXY, 101)
    px = dxy/np.mean(dxy) / np.maximum(0, cXY)
    ops['badframes'] = px > ops['th_badframes']
    ymin = np.maximum(0, np.ceil(np.amax(yoff[np.logical_not(ops['badframes'])])))
    ymax = ops['Ly'] + np.minimum(0, np.floor(np.amin(yoff)))
    xmin = np.maximum(0, np.ceil(np.amax(xoff[np.logical_not(ops['badframes'])])))
    xmax = ops['Lx'] + np.minimum(0, np.floor(np.amin(xoff)))
    ops['yrange'] = [int(ymin), int(ymax)]
    ops['xrange'] = [int(xmin), int(xmax)]
    ops['corrXY'] = corrXY

    ops['yoff'] = yoff
    ops['xoff'] = xoff

    if ops['nonrigid']:
        ops['yoff1'] = yoff1
        ops['xoff1'] = xoff1
        ops['corrXY1'] = corrXY1

    ops['meanImg'] = meanImg/ops['nframes']

    return Zreg, ops
