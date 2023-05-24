import numpy as np
import scipy.linalg as la
from moorpy.addons.mooring_configs import spread_mooring
from moorpy.addons.FWT import assign_FWT_props
from moorpy.addons.dynamic_addons import get_mean_response, get_dynamic_tension, get_qs_tension
from moorpy.addons.auxiliaries import jonswap

def evaluate_load_condition(fwt,moor_dict,Uw,TI,Hs,Tp,omegas,gamma='default',beta=0.,iters=100,tol=0.01,eval_tensions=True):
    
    # intialize mooring system
    conv,ms = spread_mooring(moor_dict, tol=tol, maxIter=iters, no_fail=True, finite_difference=False)

    # assign mooring system to FWT object
    feasible, mass, cg, M, Khs,ms_init = assign_FWT_props(fwt,ms,adjust_ballast=False)

    # get mean loads
    F_mean = fwt.get_mean_loads(Uw,Hs,Tp,gamma,beta)

    # get mean response
    X_mean,K_moor,s,T_mean,ms_mean,max_ten_id,TA,TB,conv = get_mean_response(ms_init,F_mean, 
                                                                            tol=tol, maxIter=iters, 
                                                                            no_fail=True, finite_difference=False)

    # get dynamic response
    X_std, X_wfstd, X_lfstd, RAOs, S_X, S_Xwf, S_Xlf = fwt.get_dynamic_response(K_moor,omegas,
                                                                                Uw,Hs,Tp,TI=TI,gamma=gamma,beta=beta,
                                                                                tol=tol,iters=iters,M=M,Khs=Khs)
    if eval_tensions:
        # get dynamic tension
        S_zeta = jonswap(omegas/2/np.pi,Hs,Tp,gamma)/(2*np.pi)
        T_wfstd,S_Twf,s,r_nodes,X_nodes = get_dynamic_tension(ms_mean,max_ten_id,moor_dict,omegas,S_zeta,RAOs,
                                                            tol=tol,iters=iters)

        # get quasi-static tension
        offset = np.zeros(6)
        offset[:2] = X_lfstd[:2]
        T_lfstd,s,uplift = get_qs_tension(ms,offset,max_ten_id,tol=tol,maxIter=iters,no_fail=True,finite_difference=False)
        T_std = np.sqrt(T_wfstd**2 + T_lfstd**2)

        return X_mean, X_std, X_wfstd, X_lfstd, S_X, S_Xwf, S_Xlf, s, T_mean, T_std, T_wfstd, T_lfstd, S_Twf, uplift
    else:
        return X_mean, X_std, X_wfstd, X_lfstd, S_X, S_Xwf, S_Xlf
    
def evaluate_multi_conditions():
    return 0