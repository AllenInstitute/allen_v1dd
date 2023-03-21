#Misc
import pdb,glob,fnmatch, sys
import os, time, datetime
import glob, fnmatch

#Base
import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats as st
import multiprocessing as mp

#Plot
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from matplotlib import pyplot as plt

color_names=['windows blue','red','amber','faded green','dusty purple','orange','steel blue','pink',
             'greyish','mint','clay','light cyan','forest green','pastel purple','salmon','dark brown',
             'lavender','pale green','dark red','gold','dark teal','rust','fuchsia','pale orange','cobalt blue']

color_palette = sns.xkcd_palette(color_names)
cc = sns.xkcd_palette(color_names)

#Decoding
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from multiprocessing import Pool
#Decoding Params
nProcesses = 25
nShuffles = 100

##==================================================
def decode_labels(X,Y,train_index,test_index,classifier='LDA',clabels=None,X_test=None,Y_test=None,shuffle=True,parallel=True,classifier_kws=None):
    
    nTrials, nNeurons = X.shape

    #Split data into training and test sets
    if X_test is None:
        #Training and test set are from the same time interval
        X_train = X[train_index,:]
        X_test = X[test_index,:]
        
        #Get class labels
        Y_train = Y[train_index]
        Y_test = Y[test_index]
        
    else:
        #Training and test set are from different epochs
        X_train = X
        Y_train = Y
        train_index = np.arange(len(X_train))
#         print('.',end='')
        
    #Copy training index for shuffle decoding
    train_index_sh = train_index.copy()

    #How many classes are we trying to classify?
    class_labels,nTrials_class = np.unique(Y,return_counts=True)
    nClasses = len(class_labels)

    #Initialize Classifier
    if classifier == 'LDA':
        clf = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
    elif classifier == 'SVM':
        clf = svm.LinearSVC(max_iter=1E6,penalty='l1',dual=False)
    elif classifier == 'NLSVM':
        clf = svm.NuSVC(gamma="auto")
    elif classifier == 'QDA':
        clf = QuadraticDiscriminantAnalysis()
    elif classifier == 'NearestNeighbors':
        clf = KNeighborsClassifier()
    elif classifier == 'LinearSVM':
        clf = SVC(kernel="linear", C=0.025)
    elif classifier == 'RBFSVM':
        clf = SVC(gamma=2, C=1)
    elif classifier == 'GP':
        clf = GaussianProcessClassifier(1.0 * RBF(1.0))
    elif classifier == 'DecisionTree':
        clf = DecisionTreeClassifier(max_depth=5)
    elif classifier == 'RandomForest':
        clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    elif classifier == 'NeuralNet':
        clf = MLPClassifier(alpha=1, max_iter=1000)
    elif classifier == 'NaiveBayes':
        clf = GaussianNB()
    
    #Luca's decoder 
    if (classifier == 'Euclidean_Dist') | (classifier == 'nearest_neighbor'):
        nTrials_test, nNeurons = X_test.shape
        PSTH_train = np.zeros((nClasses,nNeurons))

        #Calculate PSTH templates from training data
        for iStim, cID in enumerate(class_labels):
            pos = np.where(Y_train == cID)[0]
            PSTH_train[iStim] = np.mean(X_train[pos],axis=0)
        
        Y_hat = np.zeros((nTrials_test,),dtype=int)
        for iTrial in range(nTrials_test):
            if classifier == 'Euclidean_Dist':
                #Predict test data by taking the minimum euclidean distance
                dist = [np.sum((X_test[iTrial] - PSTH_train[iStim])**2) for iStim in range(nClasses)]
                Y_hat[iTrial] =  class_labels[np.argmin(dist)]
            else:
                #Predict test data by taking the maximum correlation between the test population vector and training PSTHs
                Rs = [np.corrcoef(PSTH_train[iStim],X_test[iTrial])[0,1] for iStim in range(nClasses)]
                Y_hat[iTrial] =  class_labels[np.argmax(Rs)]

        #Decoding Weights not implemented

    #All other classifiers
    else:
        #Fit model to the training data
        clf.fit(X_train, Y_train)

        #Predict test data
        Y_hat = clf.predict(X_test)
        Y_hat_train = clf.predict(X_train)

        #Get weights
        decoding_weights = clf.coef_

    #Calculate confusion matrix
    kfold_hits = confusion_matrix(Y_test,Y_hat,labels=clabels)
    # pdb.set_trace()

    ##===== Perform Shuffle decoding =====##
    if shuffle:
        kfold_shf = np.zeros((nShuffles,nClasses,nClasses))
        
        if parallel:
            with Pool(50) as p:
                processes = []
                #Classify with shuffled dataset
                for iS in range(nShuffles):
                    np.random.shuffle(train_index_sh)
                    Y_train_sh = Y[train_index_sh]

                    processes.append(p.apply_async(decode_labels,args=(X_train,Y_train_sh,None,None,classifier,clabels,X_test,Y_test,False,None)))

                #Extract results from parallel kfold processing
                kfold_shf = np.array([p.get()[0] for p in processes])

                #Decoding weights
                decoding_weights_shf = np.array([p.get()[2] for p in processes])

        else:
            
            kfold_shf_list = []
            decoding_weights_list = []
            for iS in range(nShuffles):
                np.random.shuffle(train_index_sh)
                Y_train_sh = Y[train_index_sh]
                kfshf, _, decoding_weights, _ = decode_labels(X_train,Y_train_sh,None,None,classifier,clabels,X_test,Y_test,False,None)
                kfold_shf_list.append(kfshf)
                decoding_weights_list.append(decoding_weights)
            kfold_shf = np.array(kfold_shf_list)
            decoding_weights_shf = np.array(decoding_weights_list)

        return kfold_hits, kfold_shf, decoding_weights, decoding_weights_shf
    else:
        return kfold_hits, [], decoding_weights, []

#     pdb.set_trace()
    # return kfold_hits, kfold_shf, decoding_weights, decoding_weights_z, decoding_weights_m_shf, decoding_weights_s_shf

##==================================================
def calculate_accuracy(results,method='L1O',plot_shuffle=False,pdfdoc=None):
    
    nClasses = results[0][0].shape[0]
    #Save results to these
    confusion_mat = np.zeros((nClasses,nClasses))
    confusion_shf = np.zeros((nClasses,nClasses))
    confusion_z = np.zeros((nClasses,nClasses))
    
    if method == 'L1O':    
        c_shf = np.zeros((nShuffles,nClasses,nClasses))
        kfold_dw = []
        kfold_dwshf = []

        for iK,rTuple in enumerate(results):
            loo_hits = rTuple[0] #size [nClasses x nClasses]
            loo_shf = rTuple[1]  #size [nShuffles,nClasses x nClasses]
            decoding_weights = rTuple[2]  #size [nClasses x nNeurons]
            decoding_weights_shf = rTuple[3]  #size [nShuffles,nClasses x nClasses]
            kfold_dw.append(rTuple[2]); kfold_dwshf.append(rTuple[3])

            #Add hits to confusion matrix
            confusion_mat += loo_hits

            #Loop through shuffles
            for iS in range(nShuffles):
                #Add hits to confusion matrix
                c_shf[iS] += loo_shf[iS]

        # pdb.set_trace()

        ##Calculate z-score of the decoding weights
        decoding_weights = np.mean(kfold_dw,axis=0)
        decoding_weights_shf = np.mean(kfold_dwshf,axis=0)
        m_shf, s_shf = np.mean(decoding_weights_shf,axis=0), np.std(decoding_weights_shf,axis=0)
        decoding_weights_z = np.divide(decoding_weights - m_shf,s_shf,np.zeros(decoding_weights.shape),where=s_shf!=0)
        
        #Calculate decoding accuracy for this leave-1-out x-validation
        confusion_mat = confusion_mat/np.sum(confusion_mat,axis=1).reshape(-1,1)

        #Loop through shuffles
        for iS in range(nShuffles):
            #Calculate shuffled decoding accuracy for this leave-1-out shuffle
            c_shf[iS] = c_shf[iS]/np.sum(c_shf[iS],axis=1).reshape(-1,1)

        #Calculate z-score 
        m_shf, s_shf = np.mean(c_shf,axis=0), np.std(c_shf,axis=0)
        confusion_shf = m_shf
        confusion_z =  (confusion_mat - m_shf)/s_shf

#         pdb.set_trace()
#         #Get signficance of decoding 
#         pvalues_loo = st.norm.sf(confusion_z)
        
        #Calculate the signifance of the unshuffled FCF results given the surrogate distribution
        N = confusion_mat.shape[0]
        pvalues = 1-2*np.abs(np.array([[st.percentileofscore(c_shf[:,i,j],confusion_mat[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)
        
        if plot_shuffle:
            #Plot shuffle distributions
            title = 'Leave-1-Out Cross-Validation'
            plot_decoding_shuffle(confusion_mat, c_shf, pvalues_loo, title,pdfdoc)
            
    elif method == 'kfold':       
        kfold_accuracies = []
        shf_accuracies = []
        kfold_zscores = []
        kfold_pvalues = []
        kfold_dw = []
        kfold_dwz = []

        #Calculate decoding accuracy per kfold
        for iK,rTuple in enumerate(results):
            kfold_hits = rTuple[0] #size [nClasses x nClasses]
            kfold_shf = rTuple[1]  #size [nShuffles,nClasses x nClasses]
            decoding_weights = rTuple[2]  #size [nClasses x nNeurons]
            decoding_weights_shf = rTuple[3]  #size [nShuffles,nClasses x nClasses]

            ##Calculate z-score of the decoding weights for this kfold
            m_shf, s_shf = np.mean(decoding_weights_shf,axis=0), np.std(decoding_weights_shf,axis=0)
            decoding_weights_z = np.divide(decoding_weights - m_shf,s_shf,np.zeros(decoding_weights.shape),where=s_shf!=0)
            kfold_dw.append(decoding_weights); kfold_dwz.append(decoding_weights_z) 
            # pdb.set_trace()

            #Normalize confusion matrix
            cm = kfold_hits/np.sum(kfold_hits,axis=1).reshape(-1,1)
            kfold_accuracies.append(cm)
            
            #Loop through shuffles and normalize
            c_shf = np.zeros((nShuffles,nClasses,nClasses))
            for iS in range(nShuffles):
                c_shf[iS] = kfold_shf[iS]/np.sum(kfold_shf[iS],axis=1).reshape(-1,1)

            #Calculate z-score of the decoding accuracy for this kfold
            m_shf, s_shf = np.mean(c_shf,axis=0), np.std(c_shf,axis=0)
            shf_accuracies.append(m_shf)
            kfold_z = np.divide(kfold_accuracies[iK] - m_shf,s_shf,np.zeros(kfold_hits.shape),where=s_shf!=0)
            kfold_zscores.append(kfold_z)

#             #Get signficance of decoding 
#             pvalues_kfold = st.norm.sf(kfold_z)
#             pdb.set_trace()

            #Calculate the signifance of the unshuffled FCF results given the surrogate distribution
            N = cm.shape[0]
            pvalues_kfold = 1-2*np.abs(np.array([[st.percentileofscore(c_shf[:,i,j],cm[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)
            kfold_pvalues.append(pvalues_kfold)
        
            if plot_shuffle:
                #Plot shuffle distributions
                title = 'Shuffle Distributions for kfold {}'.format(iK)
                plot_decoding_shuffle(kfold_accuracies[iK], c_shf, pvalues_kfold, title,pdfdoc)

        #Take average over kfolds
        confusion_mat = np.mean(kfold_accuracies,axis=0)
        confusion_shf = np.mean(shf_accuracies,axis=0)
        confusion_z = np.mean(kfold_zscores,axis=0)
        # stdevs = (np.std(kfold_accuracies,axis=0),np.std(kfold_zscores,axis=0))
        pvalues = np.mean(kfold_pvalues,axis=0)
        # decoding_weights = np.mean(kfold_dwz,axis=0)
        # decoding_weights_z = np.mean(kfold_dwz,axis=0)

        decoding_weights = np.array(kfold_dwz)
        decoding_weights_z = np.array(kfold_dwz)
        
    return confusion_mat, confusion_shf, confusion_z, pvalues, decoding_weights, decoding_weights_z
    
##==================================================
def cross_validate(X,Y,Y_sort=None,method='L1O',nKfold=5,classifier='SVM',classifier_kws=None,clabels=None,shuffle=True,plot_shuffle=False,parallel=False,nProcesses=30,pdfdoc=None):
    ##===== Description =====##
    #The main difference between these 2 methods of cross-validation are that kfold approximates the decoding accuracy per kfold 
    #and then averages across folds to get an overall decoding accuracy. This is faster and a better approximation of the actual
    #decoding accuracy if you have enough data. By contrast, Leave-1-out creates just 1 overall decoding accuracy by creating 
    #a classifier for each subset of data - 1, and adding those results to a final confusion matrix to get an estimate of the 
    #decoding accuracy. While this is what you have to do in low-data situations, the classifiers are very similar and thus 
    #share a lot of variance. Regardless, I've written both ways of calculating the decoding accuracy below. 
    
    if Y_sort is None:
        Y_sort = Y
        
    #Leave-1(per group)-out
    if method == 'L1O':
        _,nTrials_class = np.unique(Y,return_counts=True)
        k_fold = StratifiedKFold(n_splits=nTrials_class[0])
    #Or k-fold
    elif method == 'kfold':
        k_fold = StratifiedKFold(n_splits=nKfold)
            
    #Multi-processing module is weird and might hang, especially with jupyter; try without first
    if parallel:
        pool = mp.Pool(processes=nProcesses)
        processes = []
    results = []
    
    ##===== Loop over cross-validation =====##
    for iK, (train_index, test_index) in enumerate(k_fold.split(X,Y_sort)):
        # print(np.unique(Y[train_index],return_counts=True))
        # print(np.unique(Y_sort[train_index],return_counts=True))
#         pdb.set_trace()
        if parallel:
            processes.append(pool.apply_async(decode_labels,args=(X,Y,train_index,test_index,classifier,clabels,None,None,shuffle,False,classifier_kws)))
        else:
            # print(f'\nkfold {iK} - ')
            tmp = decode_labels(X,Y,train_index,test_index,classifier,clabels,None,None,shuffle,False,classifier_kws)
            results.append(tmp)
    # pdb.set_trace()
    #Extract results from parallel kfold processing
    if parallel:
        results = [p.get() for p in processes]
        pool.close()
        
    ##===== Calculate decoding accuracy =====##
    confusion_mat, confusion_shf, confusion_z, pvalues, decoding_weights, decoding_weights_z = calculate_accuracy(results,method,plot_shuffle,pdfdoc)
#     pdb.set_trace()
    return confusion_mat, confusion_shf, confusion_z, pvalues, decoding_weights, decoding_weights_z

##==============================##
##===== Plotting Functions =====##

##==================================================
# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    
##==================================================
def plot_decoding_shuffle(decoding_accuracy, shuffles, pvalues,title=None,pdfdoc=None):
    
    nClasses = decoding_accuracy.shape[-1]
    ## Plot shuffle distributions ##
    fig,axes = plt.subplots(1,nClasses,figsize=(18,6))
    plt.suptitle(title,y=1.01)

    #Plot the shuffle distribution with the mean decoding performance for that class
    for i in range(nClasses):
        ax = axes[i]
        sns.distplot(shuffles[:,i,i],color=cc[i],ax=ax)
        if pvalues[i,i] < 0.01:
            ax.set_title('element [{},{}], pval: {:.1e}'.format(i,i,pvalues[i,i]))
        else:
            ax.set_title('element [{},{}], pval: {:.2f}'.format(i,i,pvalues[i,i]))

        ax.vlines(decoding_accuracy[i,i], *ax.get_ylim(),LineWidth=2.5,label='Data: {:.2f}'.format(decoding_accuracy[i,i]))
        ax.vlines(np.mean(shuffles,axis=0)[i,i], *ax.get_ylim(),LineWidth=2.5,LineStyle = '--',label='Shuffle: {:.2f}'.format(np.mean(shuffles,axis=0)[i,i]))
        ax.set_xlim(xmin=0)
        ax.legend()

    if pdfdoc is not None:
        pdfdoc.savefig(fig)
        plt.close(fig)

##==================================================
def plot_decoding_accuracy(confusion_mat,confusion_z,ax=None,block='randomized_ctrl',class_labels=None,xylabels=None,title=None,annot=True,cmap='rocket',clims=None,cbar=True,sigfig=True,pdfdoc=None,shrink=0.5):
    #Plot decoding performance
    if ax is None:
        fig,ax = plt.subplots(figsize=(5,5))#,
        
    if title is not None:
        ax.set_title(title,fontsize=16)

    pvalues = st.norm.sf(confusion_z)
    
    if clims is None:
        clims = [np.percentile(confusion_mat,1) % 0.05, np.percentile(confusion_mat,99) - np.percentile(confusion_mat,99) % 0.05]
    #Plot actual decoding performance
    sns.heatmap(confusion_mat,annot=annot,fmt='2.2f',annot_kws={'fontsize': 10},cbar=cbar,square=True,cmap=cmap,vmin=clims[0],vmax=clims[1],cbar_kws={'shrink': shrink,'ticks':clims,'label': 'Accuracy'},ax=ax,rasterized=True)

    if sigfig:
        pval = (pvalues < 0.05) & np.eye(pvalues.shape[0],dtype=bool)
        x = np.linspace(0, pval.shape[0]-1, pval.shape[0])+0.5
        y = np.linspace(0, pval.shape[1]-1, pval.shape[1])+0.5
        X, Y = np.meshgrid(x, y)
        if len(pval) > 5:
            ax.scatter(X,Y,s=10*pval, marker='.',c='k')
        else:
            ax.scatter(X,Y,s=35*pval, marker='.',c='k')
            
#     if xylabels is not None:
#         #Labels
#         ax.set_ylabel(xylabels[0],fontsize=16)
#         ax.set_xlabel(xylabels[1],fontsize=16)
#     else:
#         ax.set_ylabel('Actual Image',fontsize=16)
#         ax.set_xlabel('Decoded Image',fontsize=16)

    if class_labels is not None:
        # ax.set_yticks(np.arange(len(class_labels))+0.5)
        # ax.set_xticks(np.arange(len(class_labels))+0.5) 
        ax.set_yticklabels(class_labels)#,va="center",fontsize=14)
        ax.set_xticklabels(class_labels)#,rotation=45)#,va="center",fontsize=14) 
    if pdfdoc is not None:
        pdfdoc.savefig(fig)
        
        
def plot_confusion_matrices(confusion_mat,confusion_z,ax=None,plot_titles=['A','B','C','D'],class_labels=['rand-pre','OB', 'Trans','rand-post']):
                                       
    nClasses = confusion_mat.shape[0]
    fig, axes = plt.subplots(1,4,figsize=(12,3))
    for i in range(nClasses):
        ax = axes[i]
        
        plot_decoding_accuracy(confusion_mat[i],confusion_z[i],ax=ax,class_labels=class_labels,xylabels=None,title=plot_titles[i],annot=False,clims=[0,1],cbar=True,sigfig=True,pdfdoc=None,shrink=0.5)
        
    return fig

        
        
                                       
                
