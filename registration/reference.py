import time, os
import numpy as np
from numpy import random as rnd
from . import bidiphase,rigid,register

def pick_initial_reference(frames):
    """ computes the initial reference image

    the seed frame is the frame with the largest correlations with other frames;
    the average of the seed frame with its top 20 correlated pairs is the
    inital reference frame returned

    Parameters
    ----------
    frames : int16
        frames from binary (frames x Ly x Lx)

    Returns
    -------
    refImg : int16
        initial reference image (Ly x Lx)

    """
    nimg,Ly,Lx = frames.shape
    frames = np.reshape(frames, (nimg,-1)).astype('float32')
    frames = frames - np.reshape(frames.mean(axis=1), (nimg, 1))
    cc = np.matmul(frames, frames.T)
    ndiag = np.sqrt(np.diag(cc))
    cc = cc / np.outer(ndiag, ndiag)
    CCsort = -np.sort(-cc, axis = 1)
    bestCC = np.mean(CCsort[:, 1:20], axis=1);
    imax = np.argmax(bestCC)
    indsort = np.argsort(-cc[imax, :])
    refImg = np.mean(frames[indsort[0:20], :], axis = 0)
    refImg = np.reshape(refImg, (Ly,Lx))
    return refImg

def iterative_alignment(ops, frames, refImg):
    """ iterative alignment of initial frames to compute reference image

    the seed frame is the frame with the largest correlations with other frames;
    the average of the seed frame with its top 20 correlated pairs is the
    inital reference frame returned

    Parameters
    ----------
    ops : dictionary
        requires 'nonrigid', 'smooth_sigma', 'bidiphase', '1Preg'

    frames : int16
        frames from binary (frames x Ly x Lx)

    refImg : int16
        initial reference image (Ly x Lx)

    Returns
    -------
    refImg : int16
        final reference image (Ly x Lx)

    """
    # do not reshift frames by bidiphase during alignment
    ops['bidiphase'] = 0
    niter = 8
    nmax  = np.minimum(100, int(frames.shape[0]/2))
    for iter in range(0,niter):
        ops['refImg'] = refImg
        maskMul, maskOffset, cfRefImg = rigid.phasecorr_reference(refImg, ops)
        freg, ymax, xmax, cmax, yxnr = register.compute_motion_and_shift(frames,
                                                    [maskMul, maskOffset, cfRefImg], ops)
        ymax = ymax.astype(np.float32)
        xmax = xmax.astype(np.float32)
        isort = np.argsort(-cmax)
        nmax = int(frames.shape[0] * (1.+iter)/(2*niter))
        refImg = freg[isort[1:nmax], :, :].mean(axis=0).squeeze().astype(np.uint16)
        dy, dx = -ymax[isort[1:nmax]].mean(), -xmax[isort[1:nmax]].mean()
        # shift data requires an array of shifts
        dy = np.array([int(np.round(dy))])
        dx = np.array([int(np.round(dx))])
        rigid.shift_data(refImg, dy, dx)
        refImg = refImg.squeeze()
        ymax, xmax = ymax+dy, xmax+dx
    return refImg

def compute_reference_image(data, ops):
    """ compute the reference image

    computes initial reference image using ops['nimg_init'] frames

    Parameters
    ----------
    data : np.array
    ops : dictionary
        requires 'nimg_init', 'nonrigid', 'smooth_sigma', 'bidiphase', '1Preg',
        'reg_file', (optional 'keep_movie_raw', 'raw_movie')

    Returns
    -------
    refImg : int16
        initial reference image (Ly x Lx)

    """

    frames = subsample_frames(data, ops)

    if ops['do_bidiphase'] and ops['bidiphase']==0:
        bidi = bidiphase.compute(frames)
        print('NOTE: estimated bidiphase offset from data: %d pixels'%bidi)
    else:
        bidi = ops['bidiphase']
    if bidi != 0:
        bidiphase.shift(frames, bidi)
    refImg = pick_initial_reference(frames)
    refImg = iterative_alignment(ops, frames, refImg)
    return refImg

def subsample_frames(data, ops):
    nFrames = ops['nframes']
    Ly = ops['Ly']
    Lx = ops['Lx']
    nsamps = int(np.minimum(ops['nimg_init'], nFrames/2))

    #Get random subset of data
    gIndices = np.arange(3,nFrames,4)
    bIndices = np.setdiff1d(np.arange(0,nFrames),gIndices)
    tmp = rnd.randint(0,nFrames,nsamps,dtype='int')
    frame_indices = np.intersect1d(tmp,bIndices)
    data_subset = data[frame_indices].copy()
    return data_subset