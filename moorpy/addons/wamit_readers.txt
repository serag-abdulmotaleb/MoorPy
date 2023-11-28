# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:59:49 2022

@author: seragela
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_hst(wamit_root, normalized = True):
    with open(wamit_root + '.hst')  as f:
            HstData = [line.strip().split() for line in f.readlines()]
        
    I = np.array([int(line[0]) for line in HstData])
    J = np.array([int(line[1]) for line in HstData])
    Cij = np.array([float(line[2]) for line in HstData])
    
    C = np.zeros([np.max(I),np.max(J)])
    for i in range(np.max(I)):
        for j in range(np.max(J)):
            C[i,j] = Cij[np.all([I==i+1,J==j+1],axis = 0)]
    return C

def read_wamit1(wamit_root, normalized = True):
    with open(wamit_root + '.1')  as f:
            RadData = [line.strip().split() for line in f.readlines()]
        
    PER = np.array([float(line[0]) for line in RadData])
    I = np.array([int(line[1]) for line in RadData])
    J = np.array([int(line[2]) for line in RadData])
    Aij = np.array([float(line[3]) for line in RadData])
    Bij = np.array([float(line[4]) for line in RadData if len(line)>4])
    
    
    periods = np.flip(np.unique(PER[PER>0]))
    
    # A = np.zeros([np.max(I),np.max(J),len(periods)])
    # B = np.zeros([np.max(I),np.max(J),len(periods)])
    # A_0 = np.zeros([np.max(I),np.max(J)])
    # A_inf = np.zeros([np.max(I),np.max(J)])
        
    A = np.zeros([len(periods),np.max(I),np.max(J)])
    B = np.zeros([len(periods),np.max(I),np.max(J)])
    A_0 = np.zeros([np.max(I),np.max(J)])
    A_inf = np.zeros([np.max(I),np.max(J)])
    
    
    for i in range(np.max(I)):
        for j in range(np.max(J)):
            if len(Aij[PER>0][np.all([I[PER>0]==i+1,J[PER>0]==j+1],axis = 0)]) > 0:
                A[:,i,j] = Aij[PER>0][np.all([I[PER>0]==i+1,J[PER>0]==j+1],axis = 0)]
            else:
                A[:,i,j] = 0
            if len(Bij[np.all([I[PER>0]==i+1,J[PER>0]==j+1],axis = 0)]) > 0:
                B[:,i,j] = Bij[np.all([I[PER>0]==i+1,J[PER>0]==j+1],axis = 0)]
            else:
                B[:,i,j] = 0
            if len(Aij[PER==0][np.all([I[PER==0]==i+1,J[PER==0]==j+1],axis = 0)]) > 0:
                A_inf[i,j] = Aij[PER==0][np.all([I[PER==0]==i+1,J[PER==0]==j+1],axis = 0)]
            else:
                A_inf[i,j] = 0
            if len(Aij[PER==-1][np.all([I[PER==-1]==i+1,J[PER==-1]==j+1],axis = 0)]) > 0:
                A_0[i,j] = Aij[PER==-1][np.all([I[PER==-1]==i+1,J[PER==-1]==j+1],axis = 0)]
            else:
                A_0[i,j] = 0
                
    if not normalized:
        A *= 1025
        A_0 *= 1025
        A_inf *= 1025
        for i in range(6):
            for j in range(6):
                B[:,i,j] *= 2*np.pi/periods*1025
        
    return periods, A, B, A_0, A_inf

def read_wamit3(wamit_root, normalized = True):
    with open(wamit_root + '.3')  as f:
        DifData = [line.strip().split() for line in f.readlines()]
    
    PER = np.array([float(line[0]) for line in DifData])
    BETA = np.array([float(line[1]) for line in DifData])
    I = np.array([int(line[2]) for line in DifData])
    ModXi = np.array([float(line[3]) for line in DifData])
    PhaXi = np.array([float(line[4]) for line in DifData])
    ReXi = np.array([float(line[5]) for line in DifData])
    ImXi = np.array([float(line[6]) for line in DifData])
    
    periods = np.flip(np.unique(PER[PER>0]))
    headings = np.unique(BETA)
    
    # Xmod = np.zeros([np.max(I),len(periods),len(headings)])
    # Xpha = np.zeros([np.max(I),len(periods),len(headings)])
    # Xre = np.zeros([np.max(I),len(periods),len(headings)])
    # Xim = np.zeros([np.max(I),len(periods),len(headings)])

    Xmod = np.zeros([len(periods),len(headings),np.max(I)])
    Xpha = np.zeros([len(periods),len(headings),np.max(I)])
    Xre = np.zeros([len(periods),len(headings),np.max(I)])
    Xim = np.zeros([len(periods),len(headings),np.max(I)])
    
    for i in range(np.max(I)):
        for m,beta in enumerate(headings):
            Xmod[:,m,i] = ModXi[np.all([I==i+1, BETA == beta],axis = 0)]
            Xpha[:,m,i] = PhaXi[np.all([I==i+1, BETA == beta],axis = 0)]
            Xre[:,m,i] = ReXi[np.all([I==i+1, BETA == beta],axis = 0)]
            Xim[:,m,i] = ImXi[np.all([I==i+1, BETA == beta],axis = 0)]
    
    if not normalized:
        Xmod *= 1025*9.81
        Xre *= 1025*9.81
        Xim *= 1025*9.81
        
    return periods, headings, Xmod, Xpha, Xre, Xim

def read_wamit5p(pressure_file,pnl_file):
    """
    Reads WAMIT's .5p and .pnl file and returns numpy arrays with the panel
    data, radiation and diffraction pressure

    Parameters
    ----------
    pressure_file: (str)
        name of the .5p file
    pnl_file: (str)
        name of the .pnl file

    Returns
    -------
    pnls: (2D numpy float array)
        An array with columns corresponding to the data in the .pnl file as.
        defined in WAMIT's manual
    p_rad: (2D numpy float array)
        An array with columns corresponding to the radiation pressure data
        in the .5p file as defined in WAMIT's manual.
    p_diff: (2D numpy float array)
        An array with columns corresponding to the radiation pressure data
        in the .5p file as defined in WAMIT's manual.
    """

    # Read .5p  and .pnl raw data
    with open(pressure_file,'r') as f:
        raw_5p = f.readlines()

    with open(pnl_file,'r') as f:
        raw_pnl = f.readlines()


    # Convert raw pressure and panel data into lists
    pnls = [line.split() for line in raw_pnl]
    p_tot = [line.split() for line in raw_5p]


    #Separate radiation and diffraction pressures
    p_rad =  []
    p_diff = []

    for row in p_tot:
        if len(row) == 15:
            p_rad.append(row)
        elif len(row) == 6:
            p_diff.append(row)


    # Convert data type from string to float and save as a numpy array
    pnls = np.asarray(pnls, dtype = np.float64)
    p_rad = np.asarray(p_rad, dtype = np.float64)
    p_diff = np.asarray(p_diff, dtype = np.float64)

    return pnls,p_rad,p_diff

def read_wamit12d(wamit_root):
    with open(wamit_root + '.12d')  as f:
            data = [line.strip().split() for line in f.readlines()]
        
    PERi = np.array([float(line[0]) for line in data])
    PERj = np.array([float(line[1]) for line in data])
    BETAi = np.array([float(line[2]) for line in data])
    BETAj = np.array([float(line[3]) for line in data])
    I = np.array([float(line[4]) for line in data])
    MODij = np.array([float(line[5]) for line in data])*1025*9.81 #TODO add normalization flag
    PHSij = np.array([float(line[6]) for line in data])
    REij = np.array([float(line[7]) for line in data])*1025*9.81 #TODO add normalization flag
    IMij = np.array([float(line[8]) for line in data])*1025*9.81 #TODO add normalization flag
    
    OMEGAi = 2*np.pi/PERi
    OMEGAj = 2*np.pi/PERj
    
    df = pd.DataFrame(columns = ['PERi','PERj','OMEGAi','OMEGAj','BETAi','BETAj','I','MODij','PHSij','REij','IMij'], 
                      data = np.array([PERi,PERj,OMEGAi,OMEGAj,BETAi,BETAj,I,MODij,PHSij,REij,IMij]).T)
    
    df = pd.concat([df,df.rename(columns={'PERi':'PERj','PERj':'PERi','OMEGAi':'OMEGAj','OMEGAj':'OMEGAi'})])
    df.drop_duplicates(inplace=True)
    
    omegasi = df['OMEGAi'].unique()
    omegasj = df['OMEGAj'].unique()
    betasi = df['BETAi'].unique()
    betasj = df['BETAj'].unique()
    
    if np.all(omegasi != omegasj):
        raise ValueError('OMEGAi and OMEGAj do not match.')
    
    if np.all(betasi != betasj):
        raise ValueError('BETAi and BETAj do not match (Multi-directional QTFs are not supported.')
    
    omegas1,omegas2,betas = np.meshgrid(omegasi,omegasj,betasi) #omegasi,omegasj,betasi
    qtf_dfs = [df[df['I'] == dof].sort_values(by = ['OMEGAi','OMEGAj','BETAi']) for dof in range(1,7)]
    qtfs = [(qtf_df['REij'] + 1j*qtf_df['IMij']).to_numpy().\
            reshape(omegasi.shape[0],omegasj.shape[0],betasi.shape[0]) for qtf_df in qtf_dfs]
        
    return df,omegas1,omegas2,betas,qtfs

