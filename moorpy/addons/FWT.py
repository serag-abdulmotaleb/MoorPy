# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 08:48:46 2023

@author: seragela
"""

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator,RegularGridInterpolator,interp1d
import scipy.stats as stats
from moorpy.addons.auxiliaries import jonswap, kaimal, kaimal_spectrum, get_transfer_function, get_linearized_damping
from moorpy.addons.gen_discon import tune_ROSCO
from moorpy.addons.wamit_readers import read_wamit1, read_wamit3, read_wamit12d
import matplotlib.pyplot as plt

def get_output_pdf(x,f_x,y,nbins=5):
    dx = np.mean(np.diff(x))
    dy = (y.max()-y.min())/nbins
    y_bins = np.arange(y.min()+dy/2,y.max()-dy/2+dy,dy)
    f_y = np.array([np.sum(f_x[(y>=y_bin-dy/2) & (y<y_bin+dy/2)])*dx/dy for y_bin in y_bins])
    y_mean = np.sum(f_y*dy*y_bins)
    y_mode = y_bins[f_y == f_y.max()]
    return y_bins,f_y,dy,y_mean,y_mode

def assign_FWT_props(FWT,ms,adjust_ballast=True):
    ms.bodyList[0].v = FWT.volDisp; 
    ms.bodyList[0].rM = np.array([0,0,FWT.cb[2]+FWT.Iwp/FWT.volDisp]); 
    ms.bodyList[0].AWP = FWT.Awp

    if adjust_ballast:
        F_v0 = ms.bodyList[0].getForces(lines_only = True) # vertical pretension
        feasible, mass, cg, M, Khs = FWT.adjust_ballast(F_v0,adjust_FWT=False)
        ms.bodyList[0].m = mass
        ms.bodyList[0].rCG = cg
    else:
        ms.bodyList[0].m = FWT.mass
        ms.bodyList[0].rCG = FWT.cg
        feasible, mass, cg, M, Khs = True, FWT.mass, FWT.cg, FWT.M, FWT.Khs

    return feasible, mass, cg, M, Khs

class FWT:
    def __init__(self,fwt_dict):
        # Platform properies
        self.mass = fwt_dict['mass'] # total fwt mass
        self.volDisp = fwt_dict['volDisp'] # fwt volume of displacement
        self.ptfmPos = np.array(fwt_dict['ptfmPosition']) # platform position displacement vector (6x1)
        self.IMom = np.array(fwt_dict['MOI']) # moments of inertia (3x1)
        self.cg = np.array(fwt_dict['CG']) # coordinates of fwt cog (3x1)
        self.cb = np.array(fwt_dict['CB']) # coordinates of fwt cob (3x1)
        self.Awp = fwt_dict['Awp'] # water plane area
        self.Iwp = fwt_dict['Iwp'] # second moment of water plane area about y axis
        self.Ab = fwt_dict['ballArea'] # horizontal area of ballast tanks available to adjust for changes in pretension
        self.hBall0 = fwt_dict['ballLvl'] # inital ballast level in ballast tanks
        self.zBot = fwt_dict['ballBot'] # vertical location of the ballast tanks bottom
        self.hBallMax = fwt_dict['ballMaxLvl'] # maximum ballast level in tanks
        self.rhoBall = fwt_dict['ballDensity'] # ballast density
        self.Blin = fwt_dict['linDamping']
        self.Bq = fwt_dict['quadDamping']
        self.wamitRoot = fwt_dict['wamitRoot']
        
        # RNA properties
        self.hHub = fwt_dict['hHub']

        # Tower properties
        z_twr = fwt_dict['TwrElev']
        d_twr = fwt_dict['TwrDiam']
        Cd_twr = fwt_dict['TwrCd']
        
        self.zTwr = z_twr
        self.dTwr = d_twr
        self.CdTwr = Cd_twr
        
        # Mass and hydrostatic stiffness matrices
        mass = self.mass
        cg = self.cg
        Imom = self.IMom

        M = np.array([[       mass,         0.0,         0.0,         0.0,  mass*cg[2], -mass*cg[1]],
                      [        0.0,        mass,         0.0, -mass*cg[2],         0.0,  mass*cg[0]],
                      [        0.0,         0.0,        mass,  mass*cg[1], -mass*cg[0],         0.0],
                      [        0.0, -mass*cg[2],  mass*cg[1],     Imom[0],         0.0,         0.0],
                      [ mass*cg[2],         0.0, -mass*cg[0],         0.0,     Imom[1],         0.0],
                      [ mass*cg[1], -mass*cg[0],         0.0,         0.0,         0.0,      Imom[2]]])
        
        self.M = M
        
        rho = 1025
        g = 9.81
        C = np.zeros([6,6])
        C[2,2] = rho*g*self.Awp
        C[3,3] = rho*g*self.Iwp + rho*g*self.volDisp*self.cb[2] - self.mass*g*self.cg[2]
        C[4,4] = rho*g*self.Iwp + rho*g*self.volDisp*self.cb[2] - self.mass*g*self.cg[2]
        
        self.Khs = C

        # Hydrodynamics
        wamit_root = fwt_dict['wamitRoot']
        
        ## 1st order coefficients
        periods1, A, B, A_0, A_inf = read_wamit1(wamit_root, normalized = False)
        periods2, betas, Xmod, Xpha, Xre, Xim = read_wamit3(wamit_root, normalized = False)
        omegas = 2*np.pi/periods1
        
        self.omegas = omegas
        self.A = A; self.B = B; self.A0 = A_0; self.Ainf = A_inf; 
        self.betas = betas; self.Xabs = Xmod; self.Xpha = Xpha; self.Xre = Xre; self.Xim = Xim;
        
        ## 2nd order coefficients TODO: transfer to separate get_qtf_functions
        qtfs_df,omegas1,omegas2,betas,qtfs = read_wamit12d(wamit_root)
        
        qtf_funcs = np.empty(6,dtype = 'object')
        
        for i in range(6):
            df = qtfs_df[qtfs_df['I'] == i+1]
            x = df['OMEGAi'].to_numpy()
            y = df['OMEGAj'].to_numpy()
            
            if np.unique(betas).shape[0] == 1:
                self.multiQTFsBetas = False
                points = (x,y)
            else:
                self.multiQTFsBetas = True
                z = df['BETAi'].to_numpy()
                points = (x,y,z)
                
            values = (df['REij'] + 1j*df['IMij']).to_numpy()
            qtf_funcs[i] = LinearNDInterpolator(points, values, fill_value = 0.0)        
           
        self.QTFsDf = qtfs_df
        self.QTFsGrid = (omegas1,omegas2,betas)
        self.QTFs = qtfs
        self.QTFsFuncs = qtf_funcs
        
        # Rotor aerodynamics and control
        cc_flag = fwt_dict['CCBlade']
        aero_root = fwt_dict['aeroRoot']
        if any(cc_flag): #not set([aero_mean,aero_exc,aero_dmp]).isdisjoin([-1]):
            yaml_file = fwt_dict['yamlFile']
            turbine, controller = tune_ROSCO(yaml_file, write_discon = False)

            ## Mean loads and gradients
            v = controller.v
            pitch_op = controller.pitch_op
            pitch_min = controller.ps_min_bld_pitch
            pitch = np.clip(pitch_op,pitch_min,None)
            omega = controller.omega_op
            
            rotor = turbine.cc_rotor
            rotor.derivatives = True
            loads,derivs = rotor.evaluate(v,omega*60/2/np.pi,pitch*180/np.pi)
            
            self.ccWindSpeed = v
            self.ccRotorSpeed = omega
            self.ccBldPitch = pitch
            self.ccPowerCrv = loads['P']
            self.ccThrustCrv = loads['T']
            self.ccTorqueCrv = loads['Q']
            self.ccController = controller
            self.ccTurbine = turbine
            self.ccRotor = rotor
            self.ccAeroOutputs = loads
            self.ccAeroDerivs = derivs

        # TODO: Generate br and ar functions if aero_root is an iterable (should be reflected in get_aero_coeffs method)
        self.aero = int(bool(fwt_dict['aerodynamics']))
        self.ccFlag = cc_flag
        self.avgAero = fwt_dict['aeroAvg']
        if cc_flag[0]:
            self.windSpeed = self.ccWindSpeed
            self.thrustCrv =  self.ccThrustCrv
        else: # from stats (csv) file
            df_thrust_avg = pd.read_csv(aero_root + '.stats',index_col=0)
            self.windSpeed = df_thrust_avg['Uw'].to_numpy()
            self.thrustCrv = df_thrust_avg['mean'].to_numpy()
        
        if not cc_flag[1]:
            aero1 = np.loadtxt(aero_root + '.1')
            U_vars = np.unique(aero1[:,0])
            f_vars = np.unique(aero1[:,1])
            x,y = np.meshgrid(U_vars,f_vars,indexing='ij')
            b_grid = aero1[:,2].reshape(len(U_vars),len(f_vars))
            a_grid = aero1[:,3].reshape(len(U_vars),len(f_vars))
            self.bAero = RegularGridInterpolator((U_vars,f_vars),b_grid,
                                 bounds_error=False, fill_value=None)
            self.aAero = RegularGridInterpolator((U_vars,f_vars),a_grid,
                                 bounds_error=False, fill_value=None)
            # self.bAero = LinearNDInterpolator(list(zip(aero1[:,0],aero1[:,1])),aero1[:,2], fill_value = 0.0)
            # self.aAero = LinearNDInterpolator(list(zip(aero1[:,0],aero1[:,1])),aero1[:,3], fill_value = 0.0)
        
        if not cc_flag[2]:
            aero2 = np.loadtxt(aero_root + '.2')
            U_vars = np.unique(aero2[:,0])
            f_vars = np.unique(aero2[:,1])
            x,y = np.meshgrid(U_vars,f_vars,indexing='ij')
            f_grid = (aero2[:,2]+ 1j*aero2[:,3]).reshape(len(U_vars),len(f_vars))
            self.fAero = RegularGridInterpolator((U_vars,f_vars),f_grid,
                                 bounds_error=False, fill_value=None)
            # self.fAero = LinearNDInterpolator(list(zip(aero2[:,0],aero2[:,1])),aero2[:,2] + 1j*aero2[:,3], fill_value = 0.0)

    def get_aero_coeffs(self,omega,Uw,sigma_u,beta = 0.):
        
        A_aero = np.zeros([6,6,len(omega)])
        B_aero = np.zeros([6,6,len(omega)])
        H_UF = np.zeros([6,len(omega)])

        # Aerodynamic transformation matrices
        h_hub = self.hHub
        r_hub = np.array([0., 0., h_hub])
                
        aero_vec3 = np.array([np.cos(beta), np.sin(beta), 0.])
        aero_vec = np.concatenate((aero_vec3, np.cross(r_hub, aero_vec3)))
        
        alt_mat = np.array([[       0., r_hub[2],-r_hub[1]],
                            [-r_hub[2],       0., r_hub[0]],
                            [ r_hub[1],-r_hub[0],       0]])
 
        aero_mat = np.zeros([6,6])
        aero_mat[:3,:3] = np.diag(aero_vec3)
        aero_mat[:3,3:] = np.diag(aero_vec3) @ alt_mat
        aero_mat[3:,:3] = aero_mat[:3,3:].T
        aero_mat[3:,3:] = np.abs(alt_mat @ np.diag(aero_vec3) @ alt_mat.T)
        if np.any(self.ccFlag[1:]):
            U_op = self.ccController.v
            
            # Peak shaving
            pitch = self.ccBldPitch # min blade pitch schedule
            
            # Blade pitch gain scheduling
            pitch_op_pc = self.ccController.pitch_op_pc 
            kp_pc = self.ccController.pc_gain_schedule.Kp
            ki_pc = self.ccController.pc_gain_schedule.Ki
            
            # Generator torque gain schedulig
            Omega_op = self.ccController.omega_op
            U_vs = self.ccController.v_below_rated
            kp_vs = self.ccController.vs_gain_schedule.Kp
            ki_vs = self.ccController.vs_gain_schedule.Ki
            KP_float = 0

            #Averaging
            if self.avgAero:
                U_vals = np.linspace(Uw-3*sigma_u,Uw+3*sigma_u,100)
                p = stats.norm.pdf(U_vals,Uw,sigma_u)*np.mean(np.diff(U_vals))
                # Operational blade pitch angle and rotor speed
                # y_list = [np.interp(U_vals,U_op,pitch),
                #           np.interp(U_vals,U_op,np.diag(self.ccAeroDerivs['dT']['dUinf'])),
                #           np.interp(U_vals,U_op,np.diag(self.ccAeroDerivs['dT']['dOmega'])) * 30/np.pi,
                #           np.interp(U_vals,U_op,np.diag(self.ccAeroDerivs['dT']['dpitch'])) * 180/np.pi,
                #           np.interp(U_vals,U_op,np.diag(self.ccAeroDerivs['dQ']['dUinf'])),
                #           np.interp(U_vals,U_op,np.diag(self.ccAeroDerivs['dQ']['dOmega'])) * 30/np.pi,
                #           np.interp(U_vals,U_op,np.diag(self.ccAeroDerivs['dQ']['dpitch'])) * 180/np.pi,]
                
                # y_modes = np.zeros(len(y_list),dtype='float')
                
                # for iy,y in enumerate(y_list):
                #     _,_,_,_,y_modes[iy] = get_output_pdf(U_vals,p,y,nbins=20)

                # pitch_i,dTdU,dTdOmega,dTdBeta,dQdU,dQdOmega,dQdBeta = y_modes

                pitch_i = np.sum(p*np.interp(U_vals,U_op,pitch))
                Omega_i = np.sum(p*np.interp(U_vals,U_op,Omega_op))

                # Load gradients
                dTdU = np.sum(p*np.interp(U_vals,U_op,np.diag(self.ccAeroDerivs['dT']['dUinf'])))
                dTdOmega = np.sum(p*np.interp(U_vals,U_op,np.diag(self.ccAeroDerivs['dT']['dOmega'])) * 30/np.pi)
                dTdBeta = np.sum(p*np.interp(U_vals,U_op,np.diag(self.ccAeroDerivs['dT']['dpitch'])) * 180/np.pi)
                
                dQdU = np.sum(p*np.interp(U_vals,U_op,np.diag(self.ccAeroDerivs['dQ']['dUinf'])))
                dQdOmega = np.sum(p*np.interp(U_vals,U_op,np.diag(self.ccAeroDerivs['dQ']['dOmega'])) * 30/np.pi)
                dQdBeta = np.sum(p*np.interp(U_vals,U_op,np.diag(self.ccAeroDerivs['dQ']['dpitch'])) * 180/np.pi)
            else:
                # Operational blade pitch angle and rotor speed
                pitch_i = np.interp(Uw,U_op,pitch)
                Omega_i = np.interp(Uw,U_op,Omega_op)

                # Load gradients
                dTdU = np.interp(Uw,U_op,np.diag(self.ccAeroDerivs['dT']['dUinf']))
                dTdOmega = np.interp(Uw,U_op,np.diag(self.ccAeroDerivs['dT']['dOmega'])) * 30/np.pi
                dTdBeta = np.interp(Uw,U_op,np.diag(self.ccAeroDerivs['dT']['dpitch'])) * 180/np.pi
                
                dQdU = np.interp(Uw,U_op,np.diag(self.ccAeroDerivs['dQ']['dUinf']))
                dQdOmega = np.interp(Uw,U_op,np.diag(self.ccAeroDerivs['dQ']['dOmega'])) * 30/np.pi
                dQdBeta = np.interp(Uw,U_op,np.diag(self.ccAeroDerivs['dQ']['dpitch'])) * 180/np.pi
                
                # Compute total derivatives
                # dBetadU = np.interp(Uw,U_op,np.gradient(self.ccBldPitch,self.ccWindSpeed))
                # dOmegadU = np.interp(Uw,U_op,np.gradient(self.ccRotorSpeed,self.ccWindSpeed))
                # dTdU += dBetadU*dTdBeta + dOmegadU*dTdOmega
                # dQdU += dBetadU*dQdBeta + dOmegadU*dQdOmega

            # Set blade pitch control gains        
            if pitch_i >= pitch_op_pc.min():
                KP_pc = -np.interp(pitch_i,pitch_op_pc,kp_pc)
                KI_pc = -np.interp(pitch_i,pitch_op_pc,ki_pc)
                # Floating feedback gain
                if self.ccController.Fl_Mode == 1:
                    KP_float = -self.ccController.Kp_float
                elif self.ccController.Fl_Mode == 2:
                    KP_float = -self.ccController.Kp_float/self.hHub
            else:
                KP_pc = 0
                KI_pc = 0
                
            # Set generator torque control gains
            if Uw < self.ccTurbine.v_rated:
                KP_vs = -kp_vs[-1]#-np.interp(Uw,U_vs,kp_vs)
                KI_vs = -ki_vs[-1]#-np.interp(Uw,U_vs,ki_vs)
            else:
                KP_vs = 0
                KI_vs = 0
            
            # Drivetrain properties
            Ir = self.ccTurbine.J
            Ng = self.ccTurbine.Ng
            
            # Aerodynamic coefficients
            H_QT = ((dTdOmega + KP_pc*dTdBeta)*1j*omega + KI_pc*dTdBeta)/\
                (omega**2*Ir + 1j*omega*(dQdOmega + KP_pc*dQdBeta - Ng*KP_vs) + KI_pc*dQdBeta - Ng*KI_vs)
            
            x_aero = (dTdU - KP_float*dTdBeta - H_QT*(dQdU - KP_float*dQdBeta))
            
            if self.ccFlag[1]:
                a_aero = np.real(1/(1j*omega)*x_aero)
                b_aero = np.real(x_aero)
                    
            if self.ccFlag[2]:
                H_Uf = (dTdU - H_QT*dQdU)
                
        if not np.all(self.ccFlag[1:]):
            if self.avgAero:
                U_vals = np.linspace(Uw-3*sigma_u,Uw+3*sigma_u,100)
                U,f = np.meshgrid(U_vals,omega/2/np.pi)
                p = stats.norm.pdf(U_vals,Uw,sigma_u)*np.mean(np.diff(U_vals))
            if not self.ccFlag[1]:
                if self.avgAero:
                    a_grid = self.aAero((U,f))
                    b_grid = self.bAero((U,f))
                    a_aero = np.sum(p*a_grid,axis = 1)
                    b_aero = np.sum(p*b_grid,axis = 1)
                else:
                    a_aero = self.aAero((Uw*np.ones(len(omega)),omega/2/np.pi))
                    b_aero = self.bAero((Uw*np.ones(len(omega)),omega/2/np.pi))

            if not self.ccFlag[2]:
                if self.avgAero:
                    f_grid = self.fAero((U,f))
                    H_Uf = np.sum(p*f_grid,axis = 1)
                else:
                    H_Uf = self.fAero((Uw*np.ones(len(omega)),omega/2/np.pi))
    
        A_aero = a_aero * np.expand_dims(aero_mat,2) * self.aero
        B_aero = b_aero * np.expand_dims(aero_mat,2) * self.aero
        H_UF = H_Uf * np.expand_dims(aero_vec,1) * self.aero
        return A_aero, B_aero, H_UF
        
    
    def get_hydro1_coeffs(self,omegas,beta = 0.):
        nb = np.argmin(abs(beta - self.betas))
        if np.abs(beta-self.betas[nb]) > 0.01*self.betas[nb]:
            print(f'Warning: Wave direction chosen {beta} is not available in wave directions. Closest available is {self.betas[nb]}')
            
        A = np.zeros([6,6,len(omegas)])
        B = np.zeros([6,6,len(omegas)])
        Xre = np.zeros([6,len(omegas)])
        Xim = np.zeros([6,len(omegas)])
        for i in range(6):
            for j in range(6):
                A[i,j] = np.interp(omegas,self.omegas,self.A[i,j],left = self.A0[i,j], right = self.Ainf[i,j])
                B[i,j] = np.interp(omegas,self.omegas,self.B[i,j])
            Xre[i] = np.interp(omegas,self.omegas,self.Xre[i,:,nb])
            Xim[i] = np.interp(omegas,self.omegas,self.Xim[i,:,nb])
        
        
        nb = np.argmin(abs(beta - self.betas))
        if np.abs(beta-self.betas[nb]) > 0.01*self.betas[nb]:
            print(f'Warning: Wave direction chosen {beta} is not available in wave directions. Closest available is {self.betas[nb]}')
        
        return A,B,Xre,Xim
    
    def get_hydro2_coeffs(self,omegas,Hs,Tp,gamma,beta = 0.):
        omega_lf = 0.05*2*np.pi #omegas.max()#
        S_sd = np.zeros([6,6,len(omegas)],dtype = 'complex')
        
        for nw,mu in enumerate(omegas):
            if mu <= omega_lf:
                S_zeta = jonswap(omegas/2/np.pi, Hm0 = Hs, Tp = Tp, gamma = gamma, normalize = False)/(2*np.pi) # generate wave elevation spectrum 
                S_zeta_mu = jonswap(omegas/2/np.pi+mu/2/np.pi, Hm0 = Hs, Tp = Tp, gamma = gamma, normalize = False)/(2*np.pi) # generate wave elevation spectrum
                
                qtf = np.zeros([6,len(omegas)],dtype = 'complex')
                for i in range(6):
                    
                    if beta not in np.unique(self.QTFsGrid[2]):
                        raise Warning('Wave direction is not specified in QTF data.')
                        
                    if self.multiQTFsBetas:
                        qtf[i,:] = self.QTFsFuncs[i](omegas,omegas+mu,beta*np.ones(omegas.shape[0]))
                    
                    else:
                        qtf[i,:] = self.QTFsFuncs[i](omegas,omegas+mu)
                    
                TT = np.array([np.outer(vec,np.conj(vec).T) for vec in qtf.T])
                TT = np.moveaxis(TT,0,-1)
                
                S_sd[:,:,nw] = 8*np.trapz(TT*S_zeta*S_zeta_mu,omegas,axis=2)
        return S_sd
    
    def adjust_ballast(self,F_v0,adjust_FWT = False):
        """
        Adjusts FWT's mass to a given mooring vertical pretension

        Parameters
        ----------
        F_v0 : float
            Total mooring vertical pretension.

        Returns
        -------
        feasible : bool
            A flag to determine whether the ballast adjustment is possible or not.

        """
        dm = (1025*self.volDisp-self.mass) - F_v0/9.81 # mass adjustment to achieve desired draft under given pretension
        dhb = dm/(self.rhoBall*self.Ab) # change of ballast level required for adjustment
        dzg = dm*(self.zBot+self.hBall0+dhb)/(self.mass+dm) # change in the vertical location of the center of gravity due to ballast adjustment
        
        mass = self.mass + dm
        cg = self.cg
        cg[2] += dzg
        IMom = self.IMom
        M = np.array([[ mass,               0.0,         0.0,         0.0,  mass*cg[2], -mass*cg[1]],
                      [  0.0,              mass,         0.0, -mass*cg[2],         0.0,  mass*cg[0]],
                      [  0.0,               0.0,        mass,  mass*cg[1], -mass*cg[0],         0.0],
                      [  0.0,       -mass*cg[2],  mass*cg[1],     IMom[0],         0.0,         0.0],
                      [ mass*cg[2],         0.0, -mass*cg[0],         0.0,     IMom[1],         0.0],
                      [ mass*cg[1], -mass*cg[0],         0.0,         0.0,         0.0,     IMom[2]]])
        Khs = self.Khs
        Khs[3,3] = 1025*9.81*self.Iwp + 1025*9.81*self.volDisp*self.cb[2] - self.mass*9.81*self.cg[2]
        Khs[4,4] = 1025*9.81*self.Iwp + 1025*9.81*self.volDisp*self.cb[2] - self.mass*9.81*self.cg[2]

        if adjust_FWT:
            self.mass = mass
            self.cg = cg
            self.M = M
            self.Khs = Khs  

        if self.hBall0+dhb >= 0 or self.hBall0+dhb <= self.hBallMax: # check if there is sufficient space for ballast adjustment
            feasible = True    
            return feasible, mass, cg, M, Khs
        else:
            feasible = False
            return feasible, mass*np.nan, cg*np.nan, M*np.nan, Khs*np.nan
                   
    def get_mean_loads(self,Uw,Hs,Tp,gamma,beta):
        """
        Evaluates mean environmental loads on a FWT.        

        Parameters
        ----------
        Uw : float
            Mean wind speed at hub height (m/s).
        Hs : float
            Significant wave height.
        Tp : float
            Peak wave period.
        gamma : float
            Wave peak factor.
        beta : float
            Coaligned direction of wind and waves.

        Returns
        -------
        F_mean : numpy array
            6 DOF mean loads vector.

        """
        # Evaluate mean environmental loads
        ## Aerodynamic loads
        ### thrust
        thrust = np.interp(Uw,self.windSpeed,self.thrustCrv) # obtain thrust value at given wind speed
        # print(thrust)
        F_thrust = thrust*np.array([           np.cos(beta),
                                               np.sin(beta),
                                                        0.0,
                                    -np.sin(beta)*self.hHub,
                                     np.cos(beta)*self.hHub,
                                                        0.0]) * self.aero #TODO: add yaw moment for peripheral tower configuration
        ### Tower drag
        rho_a = 1.225
        alpha_a = 0.14
        z_twr = self.zTwr
        d_twr = self.dTwr
        Cd_twr = self.CdTwr
        Fx_twr = 0.5*rho_a*np.trapz(d_twr*Cd_twr*Uw**2*(z_twr/self.hHub)**(2*alpha_a), z_twr)
        My_twr = 0.5*rho_a*np.trapz(d_twr*Cd_twr*Uw**2*(z_twr/self.hHub)**(2*alpha_a)*z_twr, z_twr)
        F_twr = np.array([Fx_twr*np.cos(beta),
                          Fx_twr*np.sin(beta),
                                          0.0,
                          My_twr*np.sin(beta),
                          My_twr*np.cos(beta),
                                         0.0]) * self.aero
                
        ## Hydrodynamic loads
        ### mean drift
        omegas = np.unique(self.QTFsGrid[0])
        # f_a,f_ph = slow_drift(self.QTFs,omegas = omegas, mu=0,beta=beta) # extract frequencies and drift force coefficients from platform QTFs
        if beta not in np.unique(self.QTFsGrid[2]):
            raise Warning('Wave direction is not specified in QTF data.')

        if self.multiQTFsBetas:
            f_d = np.array([self.QTFsFuncs[i](omegas,omegas,beta*np.ones(omegas.shape[0])) for i in range(6)])
        else:
            f_d = np.array([self.QTFsFuncs[i](omegas,omegas) for i in range(6)])
            
        f_drift = np.real(f_d)
        S_zeta = jonswap(omegas/2/np.pi, Hm0 = Hs, Tp = Tp, gamma = gamma, normalize = False)/(2*np.pi)
        F_drift = np.array([2*np.trapz(f_drift[dof,:]*S_zeta,omegas) for dof in range(6)])
        
        #### TODO: current drag
        
        F_mean = F_thrust + F_drift + F_twr
        return F_mean
    
    def get_dynamic_response(self, K_moor, omegas, Uw, Hs, Tp, TI = 'B', gamma = 'default', beta = 0.0, 
                             tol = 0.01,iters = 500, M = 0, Khs = 0):
        
       
        # Frequency independent coefficietns (mass, quadratic damping and hydrostatic stiffness)
        if M == 0 or Khs == 0:
            M = self.M
            Khs = self.Khs
        
        Bq = self.Bq
        Blin = self.Blin
        h_hub = self.hHub
        
        # Enviromental spectra
        S_zeta = jonswap(omegas/2/np.pi, Hm0 = Hs, Tp = Tp, gamma = gamma, normalize = False)/2/np.pi # generate wave elevation spectrum
        S_Uw = kaimal(omegas/2/np.pi, Uw, h_hub,TI = TI)/2/np.pi
        # _,S_Uw = kaimal_spectrum(omegas/2/np.pi,120., Uw, h_hub,TI = TI)
        # S_Uw = S_Uw/2/np.pi
        
        # Aerodynamic coefficients and transformation matrices
        if self.controlFlag in [0,1,2,3,5]:
            A_aero, B_aero, H_UF, S_Faero = self.get_aero_coeffs(omegas,Uw,beta)
        elif self.controlFlag in [4]:
            xi1,xi5,H_UF = self.get_aero_coeffs(omegas,Uw,beta)
            
            A_aero = np.zeros([6,6,len(omegas)])
            B_aero = np.zeros([6,6,len(omegas)])
                        
            omegas_n1 = omegas/np.sqrt(1-xi1**2)
            omegas_n5 = omegas/np.sqrt(1-xi5**2)
            
            K = Khs + K_moor
            a11_aero = K[0,0]/omegas_n1**2 - (M[0,0] + self.A0[0,0])
            a55_aero = K[4,4]/omegas_n5**2 - (M[4,4] + self.A0[4,4])
            
            b11_aero = 2*xi1*np.sqrt((M[0,0] + self.A0[0,0] + a11_aero)**2*omegas_n1**2)
            b55_aero = 2*xi5*np.sqrt((M[4,4] + self.A0[4,4] + a55_aero)**2*omegas_n5**2)
            
            # A_aero[0,0,:] = a11_aero*np.abs(np.cos(beta))
            # A_aero[1,1,:] = a11_aero*np.abs(np.sin(beta))
            # A_aero[3,3,:] = a55_aero*np.abs(np.sin(beta))
            # A_aero[4,4,:] = a55_aero*np.abs(np.cos(beta))
            
            B_aero[0,0,:] = b11_aero*np.abs(np.cos(beta))
            B_aero[1,1,:] = b11_aero*np.abs(np.sin(beta))
            B_aero[3,3,:] = b55_aero*np.abs(np.sin(beta))
            B_aero[4,4,:] = b55_aero*np.abs(np.cos(beta))
        
                                  
        # 1st order radiaton and diffraction
        nb = np.argmin(abs(beta - self.betas))
        if np.abs(beta-self.betas[nb]) > 0.01*self.betas[nb]:
            print(f'Warning: Wave direction chosen {beta} is not available in wave directions. Closest available is {self.betas[nb]}')
            
        A = np.zeros([6,6,len(omegas)])
        B = np.zeros([6,6,len(omegas)])
        Xre = np.zeros([6,len(omegas)])
        Xim = np.zeros([6,len(omegas)])
        for i in range(6):
            for j in range(6):
                A[i,j] = np.interp(omegas,self.omegas,self.A[i,j],left = self.A0[i,j], right = self.Ainf[i,j])
                B[i,j] = np.interp(omegas,self.omegas,self.B[i,j])
            Xre[i] = np.interp(omegas,self.omegas,self.Xre[i,:,nb])
            Xim[i] = np.interp(omegas,self.omegas,self.Xim[i,:,nb])
        
        
        nb = np.argmin(abs(beta - self.betas))
        if np.abs(beta-self.betas[nb]) > 0.01*self.betas[nb]:
            print(f'Warning: Wave direction chosen {beta} is not available in wave directions. Closest available is {self.betas[nb]}')
        
        # 2nd order slowly varying hydrodynamic forces
        omega_lf = 0.05*2*np.pi #omegas.max()#
        S_sd = np.zeros([6,len(omegas)])
        
        for nw,mu in enumerate(omegas):
            if mu <= omega_lf:
                
                S_zeta_mu = jonswap(omegas/2/np.pi+mu/2/np.pi, Hm0 = Hs, Tp = Tp, gamma = gamma, normalize = False)/(2*np.pi) # generate wave elevation spectrum
                
                for i in range(6):
                    
                    if beta not in np.unique(self.QTFsGrid[2]):
                        raise Warning('Wave direction is not specified in QTF data.')
                        
                    if self.multiQTFsBetas:
                        qtfs = self.QTFsFuncs[i](omegas,omegas+mu,beta*np.ones(omegas.shape[0]))
                    
                    else:
                        qtfs = self.QTFsFuncs[i](omegas,omegas+mu)
                    
                    S_sd[i,nw] = 8*np.trapz(np.abs(qtfs)**2*S_zeta*S_zeta_mu,omegas)
                    
        # Initializing loop variables
        Xd_std = 0*np.ones(6)
        S_Xwf = np.zeros([6,len(omegas)])
        S_Xsd = np.zeros([6,len(omegas)])
        S_Xlf = np.zeros([6,len(omegas)])
        S_Xdwf = np.zeros([6,len(omegas)])
        S_Xdsd = np.zeros([6,len(omegas)])
        S_Xdlf = np.zeros([6,len(omegas)])
        S_X = np.zeros([6,len(omegas)])
        S_Xd = np.zeros([6,len(omegas)])
        
        H_FX = np.zeros([6,6,len(omegas)],dtype='complex')
        H_zX = np.zeros([6,len(omegas)],dtype='complex')
        H_UX = np.zeros([6,len(omegas)],dtype='complex')

        for ni in range(iters):
            for nw,omega in enumerate(omegas):
                
                A_n = A[:,:,nw] + 0*A_aero[:,:,nw]
                B_n = B[:,:,nw] + 0*B_aero[:,:,nw] + Blin
                
                F_hydro = Xre[:,nw] + 1j*Xim[:,nw]
                F_aero = 0*H_UF[:,nw]
                
                H_FX[:,:,nw] = get_transfer_function(omega,M+A_n,B_n,Bq,Xd_std,Khs+K_moor)
                H_zX[:,nw] = np.matmul(H_FX[:,:,nw],F_hydro)
                H_UX[:,nw] = np.matmul(H_FX[:,:,nw],F_aero)
                
                S_Xwf[:,nw] = S_zeta[nw]*np.abs(H_zX[:,nw])**2
                S_Xsd[:,nw] = np.diag(np.abs(H_FX[:,:,nw]))**2*S_sd[:,nw] #np.matmul(np.abs(H_FX[:,:,nw])**2,S_sd[:,nw])
                S_Xlf[:,nw] = S_Uw[nw]*np.abs(H_UX[:,nw])**2 + S_Xsd[:,nw]
                # S_Xlf[:,nw] = np.diag(H_FX[:,:,nw] @ np.diag(S_Faero[:,nw]) @ np.conjugate(H_FX[:,:,nw]).T)
                
                S_Xdwf[:,nw] = S_zeta[nw]*np.abs(1j*omega*H_zX[:,nw])**2
                S_Xdsd[:,nw] = np.diag(np.abs(1j*omega*H_FX[:,:,nw]))**2*S_sd[:,nw] #np.matmul(np.abs(1j*omega*H_FX[:,:,nw])**2,S_sd[:,nw])
                S_Xdlf[:,nw] = S_Uw[nw]*np.abs(1j*omega*H_UX[:,nw])**2 + S_Xdsd[:,nw]
                # S_Xdlf[:,nw] = np.diag(1j*omega*H_FX[:,:,nw] @ np.diag(S_Faero[:,nw]) @ np.conjugate(1j*omega*H_FX[:,:,nw]).T)
                
                S_X[:,nw] = S_Xwf[:,nw] + S_Xlf[:,nw]
                S_Xd[:,nw] = S_Xdwf[:,nw] + S_Xdlf[:,nw]
                
            Xd_std0 = Xd_std
            X_std = np.sqrt(np.trapz(S_X, omegas, axis = 1))
            Xd_std = np.sqrt(np.trapz(S_Xd, omegas, axis = 1))

            if all(np.abs(Xd_std-Xd_std0) < tol*np.abs(Xd_std0)):
                break
        
        X_wfstd = np.sqrt(np.trapz(S_Xwf, omegas, axis = 1)) # update wave frequency motion std dev
        X_lfstd = np.sqrt(np.trapz(S_Xlf, omegas, axis = 1)) # update low frequency motion std dev  
        X_std[3:] *= 180/np.pi
        X_wfstd[3:] *= 180/np.pi
        X_lfstd[3:] *= 180/np.pi
        RAOs = H_zX
        print(ni)
        
        if Uw == 18.5:
            pass
        
        return X_std, X_wfstd, X_lfstd, RAOs, S_Xwf, S_Xlf, S_X
        
    def get_dynamic_response2(self,K_moor,omegas,Uw,Hs,Tp,TI='B',gamma = 'default',beta = 0.,
                             tol = 0.01,iters = 500, M = 0, Khs = 0):
        # Frequency independent coefficietns (mass, quadratic damping and hydrostatic stiffness)
        if M == 0 or Khs == 0:
            M = self.M
            Khs = self.Khs
        
        Bq = np.diag(self.Bq)
        B_add = self.Blin
        
        # Enviromental spectra
        h_hub = self.hHub                
        S_zeta = jonswap(omegas/2/np.pi, Hm0 = Hs, Tp = Tp, gamma = gamma, normalize = False)/2/np.pi # generate wave elevation spectrum
        # S_Uw,sigma_u = kaimal(omegas/2/np.pi, Uw, h_hub,TI = TI)/2/np.pi
        _,S_Uw,sigma_u = kaimal_spectrum(omegas/2/np.pi,120., Uw, h_hub,TI = TI)
        S_Uw = S_Uw/2/np.pi

        # Rotor aerodynamic coefficients
        A_aero, B_aero, H_UF = self.get_aero_coeffs(omegas,Uw,sigma_u,beta)
        
        # TODO: tower aerodynamic coefficients
        rho_a = 1.225
        alpha_a = 0.14
        z_twr = self.zTwr
        d_twr = self.dTwr
        Cd_twr = self.CdTwr
        
        # 1st order radiaton and diffraction coefficients
        A,B,Xre,Xim = self.get_hydro1_coeffs(omegas,beta)
        
        # 2nd order slowly varying hydrodynamic spectra
        S_sd = self.get_hydro2_coeffs(omegas,Hs,Tp,gamma,beta)
        
        # Initializating
        Xd_std = 1*np.ones(6)
        S_X = np.zeros([6,6,len(omegas)], dtype = 'complex')
        S_Xwf = np.zeros([6,6,len(omegas)], dtype = 'complex')
        S_Xlf = np.zeros([6,6,len(omegas)], dtype = 'complex')
        H_FX = np.zeros([6,6,len(omegas)],dtype='complex')
        RAOs = np.zeros([6,len(omegas)],dtype = 'complex')
        
        for ni in range(iters):
            for nw,omega in enumerate(omegas):
                # Quadratic damping linearization
                B_lin = np.diag(get_linearized_damping(Bq,Xd_std)) # Platform viscous damping
                dB1 = get_linearized_damping(0.5*rho_a*d_twr*Cd_twr, np.sqrt(Xd_std[0]**2+Xd_std[1]**2), Uw*(z_twr/h_hub)**alpha_a)
                dB5 = get_linearized_damping(0.5*rho_a*d_twr*Cd_twr*z_twr**2, np.sqrt(Xd_std[3]**2+Xd_std[4]**2)*z_twr, Uw*(z_twr/h_hub)**alpha_a)
                B1 = np.trapz(dB1,z_twr)
                B5 = np.trapz(dB5,z_twr)
                B_lin += np.diag([B1*np.abs(np.cos(beta)),B1*np.abs(np.sin(beta)),0.,B5*np.abs(np.sin(beta)),B5*np.abs(np.cos(beta)),0.])
                
                # Transfer function assembly
                A_n = A[:,:,nw] + A_aero[:,:,nw]
                B_n = B[:,:,nw] + B_aero[:,:,nw] + B_add + B_lin
                H_FX[:,:,nw] = get_transfer_function(omega,M+A_n,B_n,Khs+K_moor)

                # 1st order excitation
                F_hydro = Xre[:,nw] + 1j*Xim[:,nw]
                F_aero = H_UF[:,nw]
                RAOs[:,nw] = np.matmul(H_FX[:,:,nw],F_hydro)
                
                # Excitation spectra
                S_hydro1 = S_zeta[nw] * np.outer(F_hydro,np.conj(F_hydro).T)
                S_hydro2 = S_sd[:,:,nw]
                S_aero = S_Uw[nw] * np.outer(F_aero,np.conj(F_aero).T)
                
                # Response calculation
                S_Xwf[:,:,nw] = H_FX[:,:,nw] @ S_hydro1 @ np.conj(H_FX[:,:,nw]).T
                S_Xlf[:,:,nw] = H_FX[:,:,nw] @ S_hydro2 @ np.conj(H_FX[:,:,nw]).T +\
                                H_FX[:,:,nw] @ S_aero @ np.conj(H_FX[:,:,nw]).T
                S_X[:,:,nw] = S_Xwf[:,:,nw] + S_Xlf[:,:,nw]

            Xd_std0 = Xd_std
            Xd_std = np.sqrt(np.diag(np.trapz(omegas**2*S_X, omegas, axis=2)))
            
            if all(np.abs(Xd_std-Xd_std0) < tol*np.abs(Xd_std0)):
                break
        
        X_std = np.sqrt(np.diag(np.trapz(np.abs(S_X), omegas, axis=2)))
        X_wfstd = np.sqrt(np.diag(np.trapz(np.abs(S_Xwf), omegas, axis = 2))) # update wave frequency motion std dev
        X_lfstd = np.sqrt(np.diag(np.trapz(np.abs(S_Xlf), omegas, axis = 2))) # update low frequency motion std dev  
        X_std[3:] *= 180/np.pi
        X_wfstd[3:] *= 180/np.pi
        X_lfstd[3:] *= 180/np.pi
        
        print(ni)
        
        return X_std, X_wfstd, X_lfstd, RAOs, np.abs(np.diagonal(S_Xwf).T), np.abs(np.diagonal(S_Xlf).T), np.abs(np.diagonal(S_X).T)


                
                    

