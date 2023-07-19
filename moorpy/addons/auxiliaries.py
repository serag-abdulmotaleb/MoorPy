# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:10:26 2023

@author: seragela
"""
import numpy as np
from numpy.linalg import solve
import pandas as pd
import scipy.linalg as la
from scipy.interpolate import griddata, interp1d, interp2d, make_interp_spline
from scipy.special import modstruve, iv
from moorpy.addons.external_utils import *

def solve_motions(omega,M,B,Bq,sigma,C,F):
    H = -omega**2*M + 1j*omega*(B + np.sqrt(8/np.pi)*np.matmul(Bq,np.diag(sigma))) + C 
    X = la.solve(H, F)
    return X

def get_transfer_function(omega,M,B,C):
    # H = la.inv(-omega**2*M + 1j*omega*B + C)
    H = solve(-omega**2*M + 1j*omega*B + C,np.eye(M.shape[0]))
    return H

def mean_drift(df_qtf, betai = 0.0, betaj = 0.0): #TODO: add flag to obtain from 9 or 12
    df = df_qtf[np.isclose(df_qtf['BETAi'],betai) & np.isclose(df_qtf['BETAj'],betaj)]
    omegas = df['OMEGAi'].unique()

    f_mod = np.zeros([6,len(omegas)], dtype='float')
    f_phs = np.zeros([6,len(omegas)], dtype='float')

    for i in range(6):
        f_mod[i,:] = df[(df['OMEGAi'] == df['OMEGAj']) & np.isclose(df['I'],i+1)]['MODij']
        f_phs[i,:] = df[(df['OMEGAi'] == df['OMEGAj']) & np.isclose(df['I'],i+1)]['PHSij']

    return omegas, f_mod, f_phs

def get_newman_qtfs(omegas, f0_mod, f0_phs, mu):
    mod_func = make_interp_spline(omegas, f0_mod, axis=1)
    phs_func = make_interp_spline(omegas, f0_phs, axis=1)
    
    fsv_mod = np.zeros([6,len(omegas)], dtype='float')
    fsv_phs = np.zeros([6,len(omegas)], dtype='float')
    
    for nw,omega in enumerate(omegas):
        if omega < (omegas.max()+omegas.min())/2:
            fsv_mod[:,nw] = (mod_func(omega) + mod_func(omega+mu))/2
            fsv_phs[:,nw] = (phs_func(omega) + phs_func(omega+mu))/2
        else:
            fsv_mod[:,nw] = (mod_func(omega) + mod_func(omega-mu))/2
            fsv_phs[:,nw] = (phs_func(omega) + phs_func(omega-mu))/2
        
    return fsv_mod,fsv_phs

def get_linearized_damping(Bq,sigma,U = 0):
    # if U>0:
    def erf(x_lim):
        x = np.linspace(0,x_lim,20)
        return np.trapz(1/np.sqrt(2*np.pi)*np.exp(-0.5*x**2), x, axis=0)
    
    Blin = Bq*(np.sqrt(8/np.pi)*sigma*np.exp(-1/2*(U/sigma)**2) + 4*U*erf(U/sigma))
    # else:
    #     Blin = Bq*(np.sqrt(8/np.pi)*sigma)
    return Blin
          
def get_qtf_functions(df_qtf, beta = 'all', full_qtfs = False): #TODO: implement newman's approach
    """
    Generates interpolation functions for QTFs.
    Parameters
    ----------
    df_qtf : dataframe
        A dataframe with columns corresponding to .12 wamit file.
    beta : str, float or iterable  , optional
        Wave directions. If only one value is given then functions are generated for uni-directional waves are used, 
        if two values are given then functions are generated for bi-directional waves are used. The default is 'all'.

    Returns
    -------
    df_functions : dataframe
        A dataframe contains interpolation functions objects that has two index colums that represent the direction 
        of the two waves and 12 columns for the magnitude and phase of each dof.

    """    
    if beta == 'all':
        betas = df_qtf[['BETAi','BETAj']].value_counts().index.tolist()
    
    else:
        if isinstance(beta,(list,tuple,np.ndarray)):
            betas = [(beta[0],beta[1])]
        else:
            betas = [(beta,beta)]
    
    columns = []
    [columns.extend([f'MOD{dof}',f'PHS{dof}']) for dof in range(1,7)]
    
    index = pd.MultiIndex.from_tuples(betas, names=['BETAi', 'BETAj'])
    
    df_functions = pd.DataFrame(columns = columns, index = index)
    
    for (betai,betaj) in betas:
        df_betaij =  df_qtf[np.isclose(df_qtf['BETAi'],betai) & np.isclose(df_qtf['BETAj'],betaj)] # filter data for the required directions
        
        w1 = df_betaij ['OMEGAi'].unique()
        w2 = df_betaij ['OMEGAj'].unique()
        
        if not np.all(w1 == w2):
            raise ValueError('QTF frequencies are not identical.')
        
        omegas = w1
        dofs = df_betaij ['I'].unique().astype('int')
            
        for dof in dofs:
            df_dof = df_betaij [df_betaij['I'] == dof]
            
            qtf_mod = np.zeros([len(omegas),len(omegas)], dtype='float')
            qtf_phs = np.zeros([len(omegas),len(omegas)], dtype='float')
            
            for i,omega1 in enumerate(omegas):
                for j,omega2 in enumerate(omegas[i:]):
                    qtf_mod[i,j+i] = df_dof[(np.isclose(df_dof['OMEGAj'],omega1)) & (np.isclose(df_dof['OMEGAi'],omega2))]['MODij']
                    qtf_phs[i,j+i] = df_dof[(np.isclose(df_dof['OMEGAj'],omega1)) & (np.isclose(df_dof['OMEGAi'],omega2))]['PHSij']
                    qtf_mod[j+i,i] = qtf_mod[i,j+i]
                    qtf_phs[j+i,i] = qtf_phs[i,j+i]
        
            qtf_phs *= np.pi/180
            mod_func = interp2d(w1,w2,qtf_mod,kind = 'cubic')
            phs_func = interp2d(w1,w2,qtf_phs,kind = 'cubic')
            
            df_functions.loc[(betai,betaj),f'MOD{dof}'] = mod_func
            df_functions.loc[(betai,betaj),f'PHS{dof}'] = phs_func
        
    return df_functions 

def slow_drift(df_functions, omegas, mu, beta): #TODO: implemenet newman's approach
    """
    Evaluates frequency dependent slow drift magnitude and phase at a chosen difference frequency.

    Parameters
    ----------
    df_functions : dataframe
        (output of get_qtf_functions).
    omegas : float or array of floats
        Wave frequencies.
    mu : float
        Difference frequency.
    beta : float or iterable
        Wave directions. If only one value is given then functions are generated for uni-directional waves are used, 
        if two values are given then functions are generated for bi-directional waves are used.

    Returns
    -------
    f_mod : numpy array
        Slow drift magnitude.
    f_phs : numpy array
        Slow drift phase (in rad).

    """
    
    if isinstance(beta,(list,tuple,np.ndarray)):
        betai = beta[0]
        betaj = beta[1]
        
    else:
        betai = beta
        betaj = beta
    
    f_mod = np.zeros([6,len(omegas)])
    f_phs = np.zeros([6,len(omegas)])
    
    for dof in range(1,7):
        mod_func = df_functions.loc[(betai,betaj),f'MOD{dof}']
        phs_func = df_functions.loc[(betai,betaj),f'PHS{dof}']
        
        f_mod[dof-1,:] = np.array([mod_func(x,y) for x,y in zip(omegas, omegas + mu)]).T
        f_phs[dof-1,:] = np.array([phs_func(x,y) for x,y in zip(omegas, omegas + mu)]).T
    
    return f_mod, f_phs
        
def kaimal(f,U_hub,h_hub,TI = 'B'):

    if type(TI) == str:
        if TI.capitalize() == 'A':
            Iref = 0.16
        elif TI.capitalize() == 'B':
            Iref = 0.14
        elif TI.capitalize() == 'C':
            Iref = 0.12
        else:
            raise ValueError(f'Invalid turbulence class: {TI}')
        sigma_u = Iref*(0.75*U_hub+5.6)
    else:
        sigma_u = U_hub*TI
    
    L_u = 8.1*0.7*np.min([60,h_hub])
    S_Uw = 4*sigma_u**2*L_u/U_hub/(1+6*f*L_u/U_hub)**(5/3)

    return S_Uw,sigma_u

def kaimal_spectrum(f,R,U_hub,h_hub,TI = 'B'):
    if type(TI) == str:
        if TI.capitalize() == 'A':
            Iref = 0.16
        elif TI.capitalize() == 'B':
            Iref = 0.14
        elif TI.capitalize() == 'C':
            Iref = 0.12
        else:
            raise ValueError(f'Invalid turbulence class: {TI}')
        sigma_u = Iref*(0.75*U_hub+5.6)
    
    L_u = 8.1*0.7*np.min([60,h_hub])
    
    S_Uw = 4*sigma_u**2*L_u/U_hub/(1+6*f*L_u/U_hub)**(5/3)
    
    kappa = 12 * np.sqrt((f/U_hub)**2 + (0.12 / L_u)**2)

    Rot = (2*S_Uw / (R * kappa)**3) * \
        (modstruve(1,2*R*kappa) - iv(1,2*R*kappa) - 2/np.pi + \
            R*kappa * (-2 * modstruve(-2,2*R*kappa) + 2 * iv(2,2*R*kappa) + 1) )
            
    Rot[np.isnan(Rot)] = 0
    
    return S_Uw,Rot,sigma_u

def jonswap(f, Hm0, Tp, gamma='default', sigma_low=.07, sigma_high=.09,
            g=9.81, method='yamaguchi', normalize=True):
    '''Generate JONSWAP spectrum

    Parameters
    ----------
    f : numpy.ndarray
        Array of frequencies
    Hm0 : float, numpy.ndarray
        Required zeroth order moment wave height
    Tp : float, numpy.ndarray
        Required peak wave period
    gamma : float
        JONSWAP peak-enhancement factor (default: 3.3)
    sigma_low : float
        Sigma value for frequencies <= ``1/Tp`` (default: 0.07)
    sigma_high : float
        Sigma value for frequencies > ``1/Tp`` (default: 0.09)
    g : float
        Gravitational constant (default: 9.81)
    method : str
        Method to compute alpha (default: yamaguchi)
    normalize : bool
        Normalize resulting spectrum to match ``Hm0``

    Returns
    -------
    E : numpy.ndarray
        Array of shape ``f, Hm0.shape`` with wave energy densities

    '''

    # C Stringari - 04/06/2018
    # check input data types to avoid the following error:
    # ValueError: Integers to negative integer powers are not allowed.

    # raise an warning if the frequency array starts with zero. if the
    # user gives an array with zeros, the output will be inf at that
    # frequency
    if 0.0 in f:
        logger.warn('Frequency array contains zeros.')

    # get the input dtypes and promote to float, if needed
    f = ensure_float(f)
    Hm0 = ensure_float(Hm0)
    Tp = ensure_float(Tp)

    # Set default gamma
    if gamma == 'default':
        if Tp/np.sqrt(Hm0) <= 3.6:
            gamma = 5
        elif Tp/np.sqrt(Hm0) >= 5:
            gamma = 1
        else:
            gamma = np.exp(5.75 - 1.15*Tp/np.sqrt(Hm0))

    # check shapes of Hm0 and Tp, raise an error if the don't match
    if isinstance(Hm0, np.ndarray):
        if isinstance(Tp, np.ndarray):
            if Hm0.shape != Tp.shape:
                raise ValueError("Dimensions of Hm0 and Tp should match.")

    # This is a very naive implementation to deal with array inputs,
    # but will work if Hm0 and Tp are vectors.
    if isinstance(Hm0, np.ndarray):
        f = f[:, np.newaxis].repeat(len(Hm0), axis=1)
        Hm0 = Hm0[np.newaxis, :].repeat(len(f), axis=0)
        Tp = Tp[np.newaxis, :].repeat(len(f), axis=0)

    # Pierson-Moskowitz
    if method.lower() == 'yamaguchi':
        alpha = 1. / (.06533 * gamma ** .8015 + .13467) / 16.
    elif method.lower() == 'goda':
        alpha = 1. / (.23 + .03 * gamma - .185 / (1.9 + gamma)) / 16.
    else:
        raise ValueError('Unknown method: %s' % method)

    E_pm = alpha * Hm0**2 * Tp**-4 * f**-5 * np.exp(-1.25 * (Tp * f)**-4)

    # JONSWAP
    sigma = np.ones(f.shape) * sigma_low
    sigma[f > 1./Tp] = sigma_high

    E_js = E_pm * gamma**np.exp(-0.5 * (Tp * f - 1)**2. / sigma**2.)

    if normalize:
        # axis=0 seems to work fine with all kinds of inputs
        E_js *= Hm0**2. / (16. * trapz_and_repeat(E_js, f, axis=0))

    return E_js
