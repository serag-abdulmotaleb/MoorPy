# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:49:01 2023

@author: seragela
"""
import os
import multiprocessing
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
import copy
import numpy as np
from numpy.linalg import solve,norm
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.interpolate import interp1d
import moorpy as mp
from datetime import datetime

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
    K_moor : 2d numpy array:
        Linearized mooring stiffness matrix at mean position.
    s : numpy array
        Location of nodes along maximum tension mooring leg.
    T_mean : numpy array
        Mean tensions at node locations in maximum tension mooring leg.
    ms_mean : moorpy System object
        moorpy System object at mean position.
    TA : numpy array
        Mean line tensions at end A.
    TB : numpy array
        Mean line tensions at end B.
    conv : bool
        DESCRIPTION.
    """
    ms_mean = copy.deepcopy(ms)
    ms_mean.bodyList[0].type = 0
    ms_mean.bodyList[0].f6Ext = F_mean
    ms_mean.initialize()
    conv = ms_mean.solveEquilibrium(tol=tol, maxIter=maxIter, no_fail=no_fail, finite_difference=finite_difference)

    if conv:
        X_mean = ms_mean.bodyList[0].r6.copy()
        TA = np.array([line.TA for line in ms_mean.lineList])
        TB = np.array([line.TB for line in ms_mean.lineList])

        # tension distribution in maximum tension leg
        body = ms_mean.bodyList[0]
        fairleads = [ms_mean.pointList[pidx-1] for pidx in body.attachedP]
        fairlead_tensions = [la.norm(fairlead.getForces()) for fairlead in fairleads]
        max_ten_id = fairleads[fairlead_tensions.index(max(fairlead_tensions))].number
        leg_line_ids,point_ids = get_mooring_leg(ms_mean,max_ten_id) # get mooring leg lines and points indicies in the mooring System
        leg_lines = [ms_mean.lineList[line_id - 1] for line_id in leg_line_ids] # get lines at offset position
        leg_len = np.sum([line.L for line in leg_lines])
        n_nodes = np.sum([line.nNodes for line in leg_lines]) - (len(leg_lines) - 1)
        s = np.linspace(0,leg_len,n_nodes)
        T_mean = np.hstack([line.getLineTens()[:-1] for line in leg_lines] + [leg_lines[-1].TB]) # get tensions at offset position

    else:
        X_mean = ms_mean.bodyList[0].r6.copy()*np.nan
        TA = np.array([line.TA for line in ms_mean.lineList])*np.nan
        TB = np.array([line.TB for line in ms_mean.lineList])*np.nan
        s = np.nan
        T_mean = np.nan

    X_mean[3:]*=180/np.pi
    
    K_moor = ms_mean.bodyList[0].getStiffness()
    # K_moor = ms_mean.getSystemStiffness(DOFtype="free", dx = 0.1, dth = 0.1, solveOption=1, lines_only=True, plots=0)

    return X_mean,K_moor,s,T_mean,ms_mean,max_ten_id,TA,TB,conv

def get_mooring_leg(ms, fairlead_id):
    """    
    Finds lines indices and connection points of a multi-segmented mooring leg.
    The fuction can only handle lines that are connected in series (no branching)
    and all the lines have the same number of segments.

    Parameters
    ----------
    ms : moorpy.System
        A moorpy mooring System object.
    fairlead_id : int
        point id of the fairlead point.

    Returns
    -------
    line_ids : numpy int array
            line id number in the mooring system ordered from fairlead to anchor
    point_ids: numpy int array
            each row represents the start and end point arrays corresponding to lines in lines id ordered from fairlead to anchor.

    """
    n_fairleads = len(ms.bodyList[0].attachedP)
    n_lines = int(len(ms.lineList)/n_fairleads) # assumes that all the legs have the same number of lines
    line_ids = np.zeros(n_lines,dtype='int')
    point_ids = np.zeros([n_lines,2], dtype='int')
    
    p = ms.pointList[fairlead_id-1] # fairlead id

    for i in range(n_lines):
        line_id = p.attached[-1] # get next line's id

        line = ms.lineList[line_id - 1] # get line object
        attached_points = line.attached.copy() # get the IDs of the points attached to the line
        p1_id = p.number # get first point ID
        attached_points.remove(p1_id) # remove first point from attached point list
        p2_id = attached_points[0] # get second point id
        
        line_ids[i] = line_id
        point_ids[i,0] = p1_id
        point_ids[i,1] = p2_id
        p = ms.pointList[p2_id-1] # get second point object
        
    if p.type != 1:
        raise ValueError('Last point is not fixed.') # make sure that the last point (the anchor point) is fixed
    
    return line_ids,point_ids

def get_line_matrices(Line, LineType, sigma_u, depth, kbot, cbot, seabed_tol = 1e-3):
    """
    Evaluates mooring line dynamic equation of motion matrices.

    Parameters
    ----------
    Line : moorpy.Line object (modified package)
    LineType : moorpy.LineType object (modified package)
    sigma_u : 1D numpy array
        Standard deviation velocity vector of line nodes.
    depth : float
        Seabed depth.
    kbot : float
        Seabed vertical stiffness coefficient.
    cbot : float
        Seabed vertical damping coefficient.

    Returns
    -------
    M : 2D numpy array
        Line mass matrix.
    A : 2D numpy array
        Line added mass matrix.
    B : 2D numpy array
        Line linearized viscous damping matrix.
    K : 2D numpy array
        Line stiffness matrix.
    r_nodes : 2D numpy array
        Locations of nodes (each row represents the coords of a node)
    """
    
    n_nodes = Line.nNodes
    n_lines = n_nodes - 1 # NOTE: I am using leg->line->segment instead of line->segment->element to be consistent with moorpy's notation
    
    X_nodes,Y_nodes,Z_nodes,T_nodes = Line.getCoordinate(np.linspace(0,1,n_nodes)*Line.L) # coordinates of line nodes and tension values
    r_nodes = np.vstack((X_nodes,Y_nodes,Z_nodes)).T # coordinates of line nodes
    
    mden = LineType.mlin # line mass density function
    deq = LineType.d # line volume equivalent diameter
    Le = Line.L/n_lines # line segment (element) length
    me =  mden*Le # line segment (element) mass
    EA = LineType.EA # extensional stiffness
    Can = LineType.Can # normal added mass coeff
    Cat = LineType.Cat # tangential added mass coeff
    Cdn = LineType.Cdn # normal drag coeff
    Cdt = LineType.Cdt # tangential drag coeff
    
    M = np.zeros([3*n_nodes, 3*n_nodes], dtype='float') # mass matrix
    A = np.zeros([3*n_nodes, 3*n_nodes], dtype='float') # added mass matrix
    B = np.zeros([3*n_nodes, 3*n_nodes], dtype='float') # linearized viscous damping matrix
    K = np.zeros([3*n_nodes, 3*n_nodes], dtype='float') # stiffness matrix
    
    # Node 1 (fairlead)
    M[0:3,0:3] += me/2*np.eye(3) # element 1 mass contribution
    
    L_e2 = la.norm(r_nodes[1] - r_nodes[0]) # element 1 length
    q_e2 = (r_nodes[1] - r_nodes[0])/L_e2 # tangential unit vector
    
    sigma_uq2 = np.dot(sigma_u[0:3],q_e2) # standard deviation of tangential velocity
    sigma_un2 = np.sqrt(la.norm(sigma_u[0:3])**2 - sigma_uq2**2) # standard deviation of normal velocity
    
    Rq_e2 = np.outer(q_e2,q_e2) # local tangential to global components transformation matrix
    Rn_e2 = np.eye(3) - Rq_e2 # local normal to global components transformation matrix
    
    A_e2 = 1025*np.pi/4*deq**2*L_e2/2*(Can*Rn_e2 + Cat*Rq_e2) # element 1 added mass contribution
    B_e2 = 0.5*1025*deq*L_e2/2*np.sqrt(8/np.pi)*(Cdn*sigma_un2*Rn_e2 + 
                                                 Cdt*sigma_uq2*Rq_e2) # element 1 damping contribution
    
    K_e2 = EA/L_e2*Rq_e2 + T_nodes[0]/L_e2*Rn_e2 # element 1 stiffness (axial + geometric)
    
    A[0:3,0:3] += A_e2 
    B[0:3,0:3] += B_e2
    K[0:3,0:3] += K_e2
    K[0:3,3:6] += -K_e2
    
    if np.isclose(r_nodes[0,2],-depth,seabed_tol):
        K[2,2] += kbot
        B[2,2] += cbot 
    
    # Internal nodes loop (each internal node has contributions from two elements n-1/2 and n+1/2)
    for n in range(1, n_nodes-1):
        
        M[3*n:3*n+3,3*n:3*n+3] += me*np.eye(3) # mass contribution from adjacent elements
        
        ## element n-1/2 contributions
        L_e1 = la.norm(r_nodes[n] - r_nodes[n-1]) # element n-1/2 length
        q_e1 = (r_nodes[n] - r_nodes[n-1])/L_e1 # element n-1/2 tangential unit vector 
        sigma_uq1 = np.dot(sigma_u[3*n:3*n+3],q_e1) # standard deviation of tangential velocity
        sigma_un1 = np.sqrt(la.norm(sigma_u[3*n:3*n+3])**2 - sigma_uq1**2) # standard deviation of normal velocity
        Rq_e1 = np.outer(q_e1,q_e1)
        Rn_e1 = np.eye(3) - Rq_e1
        
        A_e1 = 1025*np.pi/4*deq**2*L_e1/2*(Can*Rn_e1 + Cat*Rq_e1) # element n-1/2 added mass contribution
        
        B_e1 = 0.5*1025*deq*L_e1/2*np.sqrt(8/np.pi)*(Cdn*sigma_un1*Rn_e1 + 
                                                     Cdt*sigma_uq1*Rq_e1) # element n-1/2 damping contribution
        
        K_e1 = EA/L_e1*Rq_e1 + T_nodes[n]/L_e1*Rn_e1
       
        ## element n+1/2 contributions
        L_e2 = la.norm(r_nodes[n+1] - r_nodes[n])
        q_e2 = (r_nodes[n+1] - r_nodes[n])/L_e2
        sigma_uq2 = np.dot(sigma_u[3*n:3*n+3],q_e2)
        sigma_un2 = np.sqrt(la.norm(sigma_u[3*n:3*n+3])**2 - sigma_uq2**2)
        Rq_e2 = np.outer(q_e2,q_e2) # local tangential to global components transformation matrix
        Rn_e2 = np.eye(3) - Rq_e2 # local normal to global components transformation matrix
        A_e2 = 1025*np.pi/4*deq**2*L_e2/2*(Can*Rn_e2 + Cat*Rq_e2) # element n+1/2 added mass contribution
        
        B_e2 = 0.5*1025*deq*L_e2/2*np.sqrt(8/np.pi)*(Cdn*sigma_un2*Rn_e2 + 
                                                     Cdt*sigma_uq2*Rq_e2) # element n-1/2 damping contribution
        
        K_e2 = EA/L_e2*Rq_e2 + T_nodes[n]/L_e2*Rn_e2
        
        ## fill line matrices
        A[3*n:3*n+3,3*n:3*n+3] += A_e1 + A_e2 # added mass
        B[3*n:3*n+3,3*n:3*n+3] += B_e1 + B_e2 # added mass
        K[3*n:3*n+3,3*n:3*n+3] += K_e1 + K_e2
        K[3*n:3*n+3,3*n-3:3*n] += -K_e1
        K[3*n:3*n+3,3*n+3:3*n+6] += -K_e2
        
        ## add seabed contribution to node n
        if np.isclose(r_nodes[n,2],-depth,1e-2):
            K[3*n+2,3*n+2] += kbot
            B[3*n+2,3*n+2] += cbot 
    
     # Node N (anchor)
    M[3*(n_nodes-1):3*(n_nodes-1)+3,3*(n_nodes-1):3*(n_nodes-1)+3] += me/2*np.eye(3) # element N-1 mass contribution
    
    L_e1 = la.norm(r_nodes[n_nodes-1] - r_nodes[n_nodes-2]) # element N-1 length
    q_e1 = 1/L_e2*(r_nodes[n_nodes-1] - r_nodes[n_nodes-2]) # tangential unit vector
    sigma_uq1 = np.dot(sigma_u[3*(n_nodes-1):3*(n_nodes-1)+3],q_e1) # standard deviation of tangential velocity 
    sigma_un1 = np.sqrt(la.norm(sigma_u[3*(n_nodes-1):3*(n_nodes-1)+3])**2 - sigma_uq1**2) # standard deviation of normal velocity
    Rq_e1 = np.outer(q_e2,q_e2) # local tangential to global components transformation matrix
    Rn_e1 = np.eye(3) - Rq_e1 # local normal to global components transformation matrix
    A_e1 = 1025*np.pi/4*deq**2*L_e1/2*(Can*Rn_e1 + Cat*Rq_e1) # element N-1 added mass contribution
    
    Bv_e1 = 0.5*1025*deq*L_e1/2*np.sqrt(8/np.pi)*(Cdn*sigma_un1*Rn_e1 + 
                                                  Cdt*sigma_uq1*Rq_e1)  # element N-1 damping contribution
    
    K_e1 = EA/L_e1*Rq_e1 + T_nodes[-1]/L_e1*Rn_e1
    
    A[3*(n_nodes-1):3*(n_nodes-1)+3,3*(n_nodes-1):3*(n_nodes-1)+3] += A_e1 # added mass
    B[3*(n_nodes-1):3*(n_nodes-1)+3,3*(n_nodes-1):3*(n_nodes-1)+3] += Bv_e1 # linearized viscous damping
    K[3*(n_nodes-1):3*(n_nodes-1)+3,3*(n_nodes-1):3*(n_nodes-1)+3] += K_e1 # stiffness matrix
    K[3*(n_nodes-1):3*(n_nodes-1)+3,3*(n_nodes-1)-3:3*(n_nodes-1)] += -K_e1 # stiffness matrix
    
    ## add seabed contribution to node N
    if np.isclose(r_nodes[n_nodes-1,2],-depth,1e-3):
        K[3*(n_nodes-1)+2,3*(n_nodes-1)+2] += kbot
        B[3*(n_nodes-1)+2,3*(n_nodes-1)+2] += cbot
    
    return M,A,B,K,r_nodes

def get_leg_matrices(ms,fairlead_id,moor_dict, sigma_u):
    """Evaluates mooring leg dynamic equation of motion matrices. A mooring leg here is defined as a serial assembly of moorpy Line objects.

    Parameters
    ----------
    ms : moorpy System object
        moorpy System object at the mean position.
    fairlead_id : int
        Index of the fairlead Point object in the moorpy Body object at which the mooring leg starts.
    moor_dict : dictionary
        dictionary with the keys specified in the spread_mooring function in the mooring_configs module.
    sigma_u : float or array
        Nodes DOFs velocity standard deviation.

    Returns
    -------
    M : 2D numpy array
        Leg mass matrix.
    A : 2D numpy array
        Leg added mass matrix.
    B : 2D numpy array
        Leg linearized viscous damping matrix.
    K : 2D numpy array
        Leg stiffness matrix.
    n_dofs: int
        Number of degrees of freedom.
    r_nodes : 2D numpy array
        Locations of nodes (each row represents the coords of a node)
    """
    
    line_ids,point_ids = get_mooring_leg(ms, fairlead_id)
    lines = [ms.lineList[line_id - 1] for line_id in line_ids]
    depth = ms.depth
    kbot = moor_dict['kbot']
    cbot = moor_dict['cbot']
    
    n_nodes = (np.sum([line.nNodes for line in lines]) - (len(lines)-1))
    n_dofs = 3*n_nodes # number of degrees of freedom is equal to total number of nodes minus the number of shared nodes x3
    M = np.zeros([n_dofs,n_dofs], dtype='float')
    A = np.zeros([n_dofs,n_dofs], dtype='float')
    B = np.zeros([n_dofs,n_dofs], dtype='float')
    K = np.zeros([n_dofs,n_dofs], dtype='float')
    r_nodes = np.zeros([n_nodes,3], dtype='float')
    n = 0
    sigma_u = np.ones(n_dofs,dtype='float')*sigma_u
    for Line in lines:
        LineType = ms.lineTypes[Line.type]
        # Filling matrices for line (dof n to dof 3xline_nodes+n)
        Mn,An,Bn,Kn,rn = get_line_matrices(Line, LineType, sigma_u[n:3*Line.nNodes+n], depth, kbot, cbot)
        M[n:3*Line.nNodes+n,n:3*Line.nNodes+n] += Mn
        A[n:3*Line.nNodes+n,n:3*Line.nNodes+n] += An
        B[n:3*Line.nNodes+n,n:3*Line.nNodes+n] += Bn
        K[n:3*Line.nNodes+n,n:3*Line.nNodes+n] += Kn
        
        M[3*Line.nNodes - 3:3*Line.nNodes, 3*Line.nNodes - 3:3*Line.nNodes] += ms.pointList[Line.attached[-1]-1].m*np.eye(3)
        n_node0 = int(n/3)
        r_nodes[n_node0:Line.nNodes+n_node0,:] = rn
        # TODO: add added mass and drag coefficient (add Cd and Ca to point object attribute)
        
        n += 3*Line.nNodes - 3 # next line starting node add the number of dofs of the current line minus 3 to get the last shared node
    
    return M,A,B,K,n_dofs,r_nodes

def get_qs_tension(ms,offset,fairlead_id, tol=0.01, maxIter=500, no_fail=True, finite_difference=False):
    """Evaluates quasi-static standard deviation in a mooring leg based on platform offset.

    Parameters
    ----------
    ms : moorpy System object
        moorpy mooring System object at mean position.
    offset : array
        6 dof offset.
    fairlead_id : int
        Index of the fairlead Point object in the moorpy Body object at which the mooring leg starts.        

    Returns
    -------
    sigma_T : array
        Quasi-static tension standard deviation at nodes.
    s : array
        Along line locations of nodes corresponding to the calculated tensions.
    uplift : bool
        Anchor uplift boolean flag.
    """
    ms_offset = copy.deepcopy(ms) # get a copy of the mooring System object
    ms_offset.bodyList[0].type = -1  # change body type to external motion input

    line_ids,point_ids = get_mooring_leg(ms_offset,fairlead_id) # get mooring leg lines and points indicies in the mooring System
    
    lines1 = [ms.lineList[line_id - 1] for line_id in line_ids] # get lines at mean position
    T1 = np.hstack([line.getLineTens()[:-1] for line in lines1] + [lines1[-1].TB]) # get tensions at mean position
    
    ms_offset.bodyList[0].r6 += offset # add offset
    ms_offset.initialize()
    conv = ms_offset.solveEquilibrium(tol=tol, maxIter=maxIter, no_fail=no_fail, finite_difference=finite_difference)

    lines2 = [ms_offset.lineList[line_id - 1] for line_id in line_ids] # get lines at offset position
    T2 = np.hstack([line.getLineTens()[:-1] for line in lines2] + [lines2[-1].TB]) # get tensions at offset position

    sigma_T = np.abs(T2-T1) # change in tension (represents qs standard deviation)
    l_bot = np.sum([line.LBot for line in lines2]) # get length of mooring leg lying on seabed
    uplift = l_bot < 1 # anchor uplift flag

    leg_len = np.sum([line.L for line in lines2])
    n_nodes = np.sum([line.nNodes for line in lines2]) - (len(lines2) - 1)
    s = np.linspace(0,leg_len,n_nodes)

    if conv:
        return sigma_T,s,uplift
    else:
        return sigma_T*np.nan,uplift*np.nan

def get_dynamic_tension(ms,fairlead_id,moor_dict,omegas,S_zeta,RAOs,tol = 0.01,iters=100, w = 0.8):
    """Evaluates dynamic tension standard deviations along a mooring leg.

    Parameters
    ----------
    ms : moorpy System object
        moorpy mooring System object at mean position.
    fairlead_id : int
        Index of the fairlead Point object in the moorpy Body object at which the mooring leg starts.        
    moor_dict : dictionary
        dictionary with the keys specified in the spread_mooring function in the mooring_configs module.
    omegas : array
        Frequencies at which to solve line dynamics in rad/s.
    S_zeta : array
        Wave elevation spectral density at specified frequencies in m^2.s/rad
    RAOs : 2d numpy array
        First order Response Ampltiude operators for 6 dofs (columns) at the specified frequencies (rows)
    tol : float, optional
        Relative tolerence for iteration convergence, by default 0.01
    iters : int, optional
        Maximum number of iterations, by default 100

    Returns
    -------
    sigma_T : array
        Dynamic tension standard deviation at nodes.
    S_T : 2d numpy array
        Dynamic tension spectra at nodes
    s : array
        Along line locations of nodes corresponding to the calculated tensions.
    r_nodes : 2d numpy array array
        Node coordinates.
    X : array
        Node dofs motion responses.
    
    """
    body = ms.bodyList[0]
    # evaluate top end motion
    fairlead = ms.pointList[fairlead_id - 1]
    r_fl = fairlead.r - body.r6[:3] #floater cg to fairlead vector
    RAO_fl = RAOs[:,:3] + np.cross(RAOs[:,3:],r_fl[np.newaxis,:],axisa=-1,axisb=-1)

    # intialize iteration matrices    
    M,A,B,K,n_dofs,r_nodes = get_leg_matrices(ms,fairlead.number,moor_dict,1.)
    X = np.zeros((len(omegas),n_dofs),dtype = 'complex')
    S_Xd = np.zeros((len(omegas),n_dofs),dtype = 'float')
    sigma_Xd = np.zeros(n_dofs,dtype = 'float')
    sigma_Xd0 = np.zeros(n_dofs,dtype = 'float')
    X[:,:3] = RAO_fl

    # solving dynamics
    start = datetime.now()
    for ni in range(iters):
        H = - omegas[:,np.newaxis,np.newaxis]**2*(M+A)[np.newaxis,:,:]\
                + 1j*omegas[:,np.newaxis,np.newaxis]*(B)[np.newaxis,:,:]\
                + K[np.newaxis,:,:]
        F = np.einsum('nij,njk->ni',-H[:,3:-3,:3],RAO_fl[:,:,np.newaxis])

        X[:,3:-3] = solve(H[:,3:-3,3:-3],F)
        S_Xd[:] = np.abs(1j*omegas[:,np.newaxis]*X)**2*S_zeta[:,np.newaxis]
        sigma_Xd[:] = np.sqrt(np.trapz(S_Xd,omegas,axis=0)) 

        if (np.abs(sigma_Xd-sigma_Xd0) <= tol*np.abs(sigma_Xd0)).all():
            break
        else:
            sigma_Xd0[:] = w * sigma_Xd + (1.-w) * sigma_Xd0
            M[:],A[:],B[:],K[:],_,_ = get_leg_matrices(ms,fairlead.number,moor_dict,sigma_Xd0)
    print(f'Finished {ni} dynamic tension iterations in {datetime.now()-start} seconds (w = {w}).')

    # evaluating motion to tension transfer function
    H_zF = np.zeros([len(omegas),n_dofs],dtype='complex')
    H_zT = np.zeros([len(omegas),int(n_dofs/3)],dtype='complex')

    # start = datetime.now()
    for nw in range(len(omegas)):
        for n in range(int(n_dofs/3 - 1)):
            H_zF[nw,3*n:3*n+3] = np.matmul(K[3*n:3*n+3,3*n+3:3*n+6],(X[nw,3*n+3:3*n+6] - X[nw,3*n:3*n+3]))
            H_zT[nw,n] = la.norm(H_zF[nw,3*n:3*n+3])
    
        H_zF[nw,-3:] = np.matmul(K[-3:,-3:],(X[nw,-6:-3] - X[nw,-3:]))
        H_zT[nw,-1] = la.norm(H_zF[nw,-3:])
    # print(f'Finished dynamic tension calculations in {datetime.now()-start} seconds.')
    
    # evaluate tension results
    S_T = np.abs(H_zT)**2*S_zeta[:,np.newaxis]
    sigma_T = np.sqrt(np.trapz(S_T,omegas,axis=0))
    
    dr = np.diff(r_nodes,axis=0)
    ds = la.norm(dr,axis=1)
    s = np.zeros(len(sigma_T))
    s[1:] = np.cumsum(ds)

    return sigma_T,S_T,s,r_nodes,X

def animate_line_motion(omega,omegas,r_nodes,X_nodes,Amp=1.,t_sim=60,fps=20):
    x_0 = r_nodes[:,0]
    y_0 = r_nodes[:,1]
    z_0 = r_nodes[:,2]
    time = np.linspace(0, t_sim, int(fps*t_sim))
    X_func = interp1d(omegas,X_nodes)
    X_omega = X_func(omega)
    fig,ax = plt.subplots(1,1,subplot_kw={"projection": "3d"})
    line = ax.plot(x_0,y_0,z_0,color='black',marker='o', markersize = 3)[0]
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    r_0 = np.sqrt(x_0**2 + y_0**2)
    ax.set_xlim(r_0.min(),r_0.max())
    ax.set_ylim(r_0.min(),r_0.max())
    ax.set_ylim(z_0.min(),z_0.max())
    fig.suptitle(f'Wave period {2*np.pi/omega} s, Wave ampliude {Amp} m')
    
    def update_frame(frn,time,line,Amp,X_omega):
        t = time[frn]
        x = x_0 + [np.real(Amp*np.abs(X_omega[i])*np.exp(1j*(omega*t+np.angle(X_omega[i])))) for i in range(0,len(X_omega),3)]
        y = y_0 + [np.real(Amp*np.abs(X_omega[i])*np.exp(1j*(omega*t+np.angle(X_omega[i])))) for i in range(1,len(X_omega),3)]
        z = z_0 + [np.real(Amp*np.abs(X_omega[i])*np.exp(1j*(omega*t+np.angle(X_omega[i])))) for i in range(2,len(X_omega),3)]
        line.set_data_3d(x,y,z)
        return line
        
    line_animation = animation.FuncAnimation(fig, update_frame, repeat=True, fargs=(time,line,Amp,X_omega),
                                             frames=int(fps*t_sim), interval=1/fps)
    
    return line_animation

def get_leg_modes(ms,fairlead_id,moor_dict):
    fairlead = ms.pointList[fairlead_id - 1]
    M,A,B,K,n_dofs,r_nodes = get_leg_matrices(ms,fairlead.number,moor_dict,0.)

    eigvals,eigvecs = la.eig(K,M)
    stable_eigvals = eigvals[np.real(eigvals)>0]
    stable_eigvecs = eigvecs[:,np.real(eigvals)>0]
    
    idx = stable_eigvals.argsort()[::-1]   
    stable_eigvals = stable_eigvals[idx]
    stable_eigvecs = stable_eigvecs[:,idx]
   
    freqs = np.sqrt(np.real(stable_eigvals))/2/np.pi
    mode_shapes = np.zeros(stable_eigvecs.shape,dtype='float')    
    
    for i in range(stable_eigvecs.shape[1]):
        mode_shapes[:,i] = r_nodes.flatten('C') + stable_eigvecs[:,i]

    return freqs,mode_shapes

def plot_mode_shape(idx,freqs,mode_shapes,r_0,amp_factor = 10.):
    r = mode_shapes[:,idx].reshape([int(len(mode_shapes[:,idx])/3),3])
    r = (r-r_0)*amp_factor
    x = r[:,0]
    y = r[:,1]
    z = r[:,2]
    
    x_0 = r_0[:,0]
    y_0 = r_0[:,1]
    z_0 = r_0[:,2]
        
    fig,ax = plt.subplots(1,1,subplot_kw={"projection": "3d"})
    ax.plot(x_0,y_0,z_0,'-ko',label='initial')
    ax.plot(x+x_0,y+y_0,z+z_0,'--ro',label='mode shape')
    fig.suptitle(f'frequency {freqs[idx]} Hz')