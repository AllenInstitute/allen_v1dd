#Base
import numpy as np
import scipy.stats as st
import ray

#For progress bar
from asyncio import Event
from typing import Tuple
from time import sleep

# #CCM
# from delay_embedding import surrogate as S
# from delay_embedding import helpers as H

import ray
# For typing purposes
from ray.actor import ActorHandle

from tqdm import tqdm



@ray.remote
def correlation_FC(X, pba=None, transform='fisher'):
    
    N, T = X.shape
            
    #Loop over each pair and calculate the correlation between signals i and j
    correlation_mat  = np.zeros((N,N))*np.nan
    for i in range(N):
        for j in range(i,N):
            if i == j:
                continue
            indy = np.where((~np.isnan(X[i])) & (~np.isnan(X[j])))[0]
            cc = np.corrcoef(X[i,indy],X[j,indy])[0,1]
            
            correlation_mat[i,j] = cc
            correlation_mat[j,i] = cc
    
    #Update progress bar if provided
    if pba is not None:
        pba.update.remote(1)
            
#     #Apply transformation   
#     if transform == 'fisher':
#         correlation_mat = np.arctanh(correlation_mat)
        
    return correlation_mat

##===== Calculate correlation & shuffle distribution ======##
def calculate_correlation(data_array, running_mask, shf_method, pba_tuple = None, nShuffles = 100,shuffle = True):
    N, T = data_array.shape

    FC = np.zeros((2,N,N))*np.nan
    if shuffle:
        pval = np.zeros((2,N,N))*np.nan
        FC_shf = np.zeros((2,nShuffles,N,N))*np.nan
    else:
        pval = []
        FC_shf = []

    if pba_tuple is not None:
        prog_bar, pba = pba_tuple
    else:
        pba = None

    tasks_pre_launch = []
    for iR, rstr in enumerate(['Rest','Running']):
        indy = np.where(running_mask == iR)[0]

        frac = np.sum(running_mask == iR)/running_mask.shape[0]
        if frac < 0.15:
            # print(f'Not enough {rstr} timepoints to calculate correlation')
            continue

        X = data_array[:,indy]

        tasks_pre_launch.append(correlation_FC.remote(X,pba)) 
        # FC[iR] = ray.get(correlation_FC.remote(X,pba))

        if shuffle:
            X_orig = X.copy()

            if shf_method == 'shuffle':
                #Calculate shuffles
                for iShf in range(nShuffles):
                    X_shf = np.apply_along_axis(np.random.permutation, 1, X_orig)
                    tasks_pre_launch.append(correlation_FC.remote(X_shf,pba)) 
            elif shf_method == 'surrogate':

                delay_vectors = np.concatenate(list(map(lambda x: H.create_delay_vector(x,1,10)[:,:,np.newaxis], X)),2)

                refs = [S.twin_surrogates.remote(delay_vectors[:,:,i],N=nShuffles) for i in range(delay_vectors.shape[2])]
                surrogates = np.array(ray.get(refs))
                for iShf in range(nShuffles):
                    tasks_pre_launch.append(correlation_FC.remote(surrogates[:,iShf],pba)) 

    #Print progress bar
    if pba_tuple is not None:
        prog_bar.print_until_done()
    FC_list = ray.get(tasks_pre_launch)

    #Extract parallel results
    counter = 0
    
    for iR, rstr in enumerate(['Rest','Running']):
        indy = np.where(running_mask == iR)[0]

        frac = np.sum(running_mask == iR)/running_mask.shape[0]
        if frac < 0.15:
            continue
        FC[iR] = FC_list[counter]
        counter += 1
        if shuffle:
            FC_shf[iR] = FC_list[counter:(counter+100)]
            counter += 100

            pval[iR] = 1-2*np.abs(np.array([[st.percentileofscore(FC_shf[iR,:,i,j],FC[iR,i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)

    return FC, FC_shf, pval


##===== Progress bar for parallel ray functions =====##
@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter
    

# Back on the local node, once you launch your remote Ray tasks, call
# `print_until_done`, which will feed everything back into a `tqdm` counter.
class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return