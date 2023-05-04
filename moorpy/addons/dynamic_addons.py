# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:49:01 2023

@author: seragela
"""
import numpy as np
import moorpy as mp

def get_mean_response(ms,F_mean, tol=0.01, maxIter=500, no_fail=True, finite_difference=False):
    """
    Evalulates mean displacements and line tensions of a mooring system given mean envionmetal loads.

    Parameters
    ----------
    ms : moorpy.System object
        A moorpy mooring system.
    F_mean : numpy array
        6 DOF mean environmental load vector.

    Returns
    -------
    X : numpy array
        Mean platform displacement.
    TA : numpy array
        Mean line tensions at end A.
    TB : numpy array
        Mean line tensions at end B.
    conv : bool
        DESCRIPTION.

    """
    ms.bodyList[0].type = 0
    ms.bodyList[0].f6Ext = F_mean
    ms.initialize()
    conv = ms.solveEquilibrium(tol=tol, maxIter=maxIter, no_fail=no_fail, finite_difference=finite_difference)

    if conv:
        X = ms.bodyList[0].r6.copy()
        TA = np.array([line.TA for line in ms.lineList])
        TB = np.array([line.TB for line in ms.lineList])
        
    else:
        X = ms.bodyList[0].r6.copy()*np.nan
        TA = np.array([line.TA for line in ms.lineList])*np.nan
        TB = np.array([line.TB for line in ms.lineList])*np.nan
    X[3:]*=180/np.pi
    
    K_moor = ms.getSystemStiffness(lines_only=True,solveOption=1,dth=0.1,dx=0.1)
    
    return X,TA,TB,K_moor,conv


# def get_dynamic_tension(ms,omegas,RAOs,S_X, method = 1):


