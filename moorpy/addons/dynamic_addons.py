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

def get_mean_response(ms, F_mean, fairlead='maxten', tol=0.01, maxIter=500, no_fail=True, finite_difference=False):
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
        if fairlead == 'maxten':
            fairlead_id = fairleads[fairlead_tensions.index(max(fairlead_tensions))].number
        else:
            fairlead_id = fairleads[fairlead-1].number
        leg_line_ids,point_ids = get_mooring_leg(ms_mean,fairlead_id) # get mooring leg lines and points indicies in the mooring System
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
    # print(fairlead_id)

    return X_mean,K_moor,s,T_mean,ms_mean,fairlead_id,TA,TB,conv

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

def get_horizontal_oop_vec(p1,p2):
    hor_vec = p2 - np.array([p1[0],p1[1],p2[2]])
    ver_vec = p1 - np.array([p1[0],p1[1],p2[2]])

    if np.isclose(la.norm(hor_vec),0): # vertical line
        n_op = np.array([1,0,0]) 
    elif np.isclose(la.norm(ver_vec),0): # horizontal line
        oop_vec = np.cross(hor_vec,np.array([0,0,1])) 
        n_op = oop_vec/la.norm(oop_vec)
    else:
        oop_vec = np.cross(hor_vec,ver_vec)
        n_op = oop_vec/la.norm(oop_vec)
    return n_op

def get_line_matrices(Line,LineType,omegas,S_zeta,r_dynamic,depth,kbot,cbot,seabed_tol=1e-4):
    """
    Evaluates mooring line dynamic equation of motion matrices.
    (Nodes are numbered 0 to N and elements are numbered 1 to N)

    Parameters
    ----------
    Line : moorpy.Line object (modified package)
    LineType : moorpy.LineType object (modified package)
    sigma_u : 1D numpy array #TODO correct input description
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
    r_mean : 2D numpy array
        Locations of nodes (each row represents the coords of a node)
    """

    # extract line properties
    N = Line.nNodes
    mden = LineType.mlin # line mass density function
    deq = LineType.d # line volume equivalent diameter
    EA = LineType.EA # extensional stiffness
    Can = LineType.Can # normal added mass coeff
    Cat = LineType.Cat # tangential added mass coeff
    Cdn = LineType.Cdn # normal drag coeff
    Cdt = LineType.Cdt # tangential drag coeff

    # extract node coordinates
    X_mean,Y_mean,Z_mean,T_mean = Line.getCoordinate(np.linspace(0,1,N)*Line.L) # coordinates of line nodes and tension values
    r_mean = np.vstack((X_mean,Y_mean,Z_mean)).T # coordinates of line nodes

    # evaluate node velocities
    v_dynamic = 1j*omegas[:,np.newaxis,np.newaxis]*r_dynamic

    # define out of plane normal
    h_op = get_horizontal_oop_vec(r_mean[0],r_mean[-1]) # horizontal out-of-plane vector
    hh_op = np.outer(h_op,h_op)

    # intialize line matrices
    M = np.zeros([3*N, 3*N], dtype='float') # mass matrix
    A = np.zeros([3*N, 3*N], dtype='float') # added mass matrix
    B = np.zeros([3*N, 3*N], dtype='float') # linearized viscous damping matrix
    K = np.zeros([3*N, 3*N], dtype='float') # stiffness matrix
    
    # Node 0 (fairlead)
    dr_e1 = r_mean[1] - r_mean[0]
    L_e1 = la.norm(dr_e1) # element 1 length
    t_e1 = (dr_e1)/L_e1 # tangential unit vector
    p_e1 = np.cross(t_e1,h_op) # in plane normal unit vector


    ut_e1 = np.einsum('ij,j->i',v_dynamic[:,0,:],t_e1) # tangential velocity
    uh_e1 = np.einsum('ij,j->i',v_dynamic[:,0,:],h_op) # normal horizontal out of plane velocity
    up_e1 = np.einsum('ij,j->i',v_dynamic[:,0,:],p_e1) # normal in plane velocity

    sigma_ut_e1 = np.sqrt(np.trapz(np.abs(ut_e1)**2*S_zeta,omegas))
    sigma_uh_e1 = np.sqrt(np.trapz(np.abs(uh_e1)**2*S_zeta,omegas))
    sigma_up_e1 = np.sqrt(np.trapz(np.abs(up_e1)**2*S_zeta,omegas))

    tt_e1 = np.outer(t_e1,t_e1) # local tangential to global components transformation matrix
    pp_e1 = np.outer(p_e1,p_e1) # local normal inplane to global components transformation matrix

    M[0:3,0:3] += mden*L_e1/2*np.eye(3) # element 1 mass contribution

    A_e1 = 1025*np.pi/4*deq**2*L_e1/2*(Can*(hh_op+pp_e1) + Cat*tt_e1) # element 1 added mass contribution

    B_e1 = 0.5*1025*deq*L_e1/2*np.sqrt(8/np.pi)*(Cdn*(sigma_uh_e1*hh_op + sigma_up_e1*pp_e1) +
                                                 Cdt*sigma_ut_e1*tt_e1) # element 1 damping contribution 

    K_e1 = EA/L_e1*tt_e1 + (T_mean[0]/L_e1)*(hh_op+pp_e1) # element 1 stiffness (axial + geometric)
    
    ## assembling element 1 contributions (rows corresponding to node 0)
    A[0:3,0:3] += A_e1 
    B[0:3,0:3] += B_e1
    K[0:3,0:3] += K_e1
    K[0:3,3:6] += -K_e1
    
    ## add seabed contribution to node 0
    if np.isclose(r_mean[0,2],-depth,seabed_tol):
        K[2,2] += kbot
        B[2,2] += cbot 

    # Internal nodes loop (each internal node has contributions from two elements n-1/2 and n+1/2)
    for n in range(1, N-1):
        
        ## backward element (n-1/2) contributions
        dr_bw = r_mean[n-1] - r_mean[n]
        L_bw = la.norm(dr_bw) # element 1 length
        t_bw = (dr_bw)/L_bw # tangential unit vector
        p_bw = np.cross(t_bw,h_op) # in plane normal unit vector

        ut_bw = np.einsum('ij,j->i',v_dynamic[:,n,:],t_bw) # tangential velocity
        uh_bw = np.einsum('ij,j->i',v_dynamic[:,n,:],h_op) # normal horizontal out of plane velocity
        up_bw = np.einsum('ij,j->i',v_dynamic[:,n,:],p_bw) # normal in plane velocity

        sigma_ut_bw = np.sqrt(np.trapz(np.abs(ut_bw)**2*S_zeta,omegas))
        sigma_uh_bw = np.sqrt(np.trapz(np.abs(uh_bw)**2*S_zeta,omegas))
        sigma_up_bw = np.sqrt(np.trapz(np.abs(up_bw)**2*S_zeta,omegas))

        tt_bw = np.outer(t_bw,t_bw) # local tangential to global components transformation matrix
        pp_bw = np.outer(p_bw,p_bw) # local normal inplane to global components transformation matrix

        M[3*n:3*n+3,3*n:3*n+3] += mden*L_bw/2*np.eye(3) # mass contribution from adjacent elements

        A_bw = 1025*np.pi/4*deq**2*L_bw/2*(Can*(hh_op+pp_bw) + Cat*tt_bw) # backward element added mass contribution

        B_bw = 0.5*1025*deq*L_bw/2*np.sqrt(8/np.pi)*(Cdn*(sigma_uh_bw*hh_op + sigma_up_bw*pp_bw) +
                                                     Cdt*sigma_ut_bw*tt_bw) # backward element damping contribution 

        K_bw = EA/L_bw*tt_bw + (T_mean[n]/L_bw)*(hh_op+pp_bw) # backward element stiffness (axial + geometric)

        ## forward element (n+1/2) contributions
        dr_fw = r_mean[n+1] - r_mean[n]
        L_fw = la.norm(dr_fw) # element 1 length
        t_fw = (dr_fw)/L_fw # tangential unit vector
        p_fw = np.cross(t_fw,h_op) # in plane normal unit vector


        ut_fw = np.einsum('ij,j->i',v_dynamic[:,n,:],t_fw) # tangential velocity
        uh_fw = np.einsum('ij,j->i',v_dynamic[:,n,:],h_op) # normal horizontal out of plane velocity
        up_fw = np.einsum('ij,j->i',v_dynamic[:,n,:],p_fw) # normal in plane velocity

        sigma_ut_fw = np.sqrt(np.trapz(np.abs(ut_fw)**2*S_zeta,omegas))
        sigma_uh_fw = np.sqrt(np.trapz(np.abs(uh_fw)**2*S_zeta,omegas))
        sigma_up_fw = np.sqrt(np.trapz(np.abs(up_fw)**2*S_zeta,omegas))

        tt_fw = np.outer(t_fw,t_fw) # local tangential to global components transformation matrix
        pp_fw = np.outer(p_fw,p_fw) # local normal inplane to global components transformation matrix
        
        M[3*n:3*n+3,3*n:3*n+3] += mden*L_fw/2*np.eye(3) # mass contribution from adjacent elements

        A_fw = 1025*np.pi/4*deq**2*L_fw/2*(Can*(hh_op+pp_fw) + Cat*tt_fw) # backward element added mass contribution

        B_fw = 0.5*1025*deq*L_fw/2*np.sqrt(8/np.pi)*(Cdn*(sigma_uh_fw*hh_op + sigma_up_fw*pp_fw) +
                                                     Cdt*sigma_ut_fw*tt_fw) # backward element damping contribution 

        K_fw = EA/L_fw*tt_fw + (T_mean[n]/L_fw)*(hh_op+pp_fw) # backward element stiffness (axial + geometric)

        ## assembling bwd and fwd elements contributions (rows corresponding to node n)
        A[3*n:3*n+3,3*n:3*n+3] += A_bw + A_fw
        B[3*n:3*n+3,3*n:3*n+3] += B_bw + B_fw
        K[3*n:3*n+3,3*n:3*n+3] += K_bw + K_fw 
        K[3*n:3*n+3,3*n-3:3*n] += -K_bw
        K[3*n:3*n+3,3*n+3:3*n+6] += -K_fw
        
        ## add seabed contribution to node n
        if np.isclose(r_mean[n,2],-depth,seabed_tol):
            K[3*n+2,3*n+2] += kbot
            B[3*n+2,3*n+2] += cbot

    # Node N (anchor)
    dr_eN = r_mean[N-1] - r_mean[N-2]
    L_eN = la.norm(dr_eN) # element 1 length
    t_eN = (dr_eN)/L_eN # tangential unit vector
    p_eN = np.cross(t_eN,h_op) # in plane normal unit vector

    ut_eN = np.einsum('ij,j->i',v_dynamic[:,N-1,:],t_eN) # tangential velocity
    uh_eN = np.einsum('ij,j->i',v_dynamic[:,N-1,:],h_op) # normal horizontal out of plane velocity
    up_eN = np.einsum('ij,j->i',v_dynamic[:,N-1,:],p_eN) # normal in plane velocity

    sigma_ut_eN = np.sqrt(np.trapz(np.abs(ut_eN)**2*S_zeta,omegas))
    sigma_uh_eN = np.sqrt(np.trapz(np.abs(uh_eN)**2*S_zeta,omegas))
    sigma_up_eN = np.sqrt(np.trapz(np.abs(up_eN)**2*S_zeta,omegas))

    tt_eN = np.outer(t_eN,t_eN) # local tangential to global components transformation matrix
    pp_eN = np.outer(p_eN,p_eN) # local normal inplane to global components transformation matrix

    M[3*(N-1):3*(N-1)+3,3*(N-1):3*(N-1)+3] += mden*L_eN/2*np.eye(3) # element 1 mass contribution

    A_eN = 1025*np.pi/4*deq**2*L_eN/2*(Can*(hh_op+pp_eN) + Cat*tt_eN) # element 1 added mass contribution

    B_eN = 0.5*1025*deq*L_eN/2*np.sqrt(8/np.pi)*(Cdn*(sigma_uh_eN*hh_op + sigma_up_eN*pp_eN) +
                                                 Cdt*sigma_ut_eN*tt_eN) # element 1 damping contribution 

    K_eN = EA/L_eN*tt_eN + (T_mean[N-1]/L_eN)*(hh_op+pp_eN) # element 1 stiffness (axial + geometric)
    
    ## assembling element 1 contributions (rows corresponding to node 0)
    A[3*(N-1):3*(N-1)+3,3*(N-1):3*(N-1)+3] += A_eN 
    B[3*(N-1):3*(N-1)+3,3*(N-1):3*(N-1)+3] += B_eN 
    K[3*(N-1):3*(N-1)+3,3*(N-1):3*(N-1)+3] += K_eN
    K[3*(N-1):3*(N-1)+3,3*(N-1)-3:3*(N-1)] += -K_eN
    
    ## add seabed contribution to node N
    if np.isclose(r_mean[N-1,2],-depth,seabed_tol):
        K[3*(N-1)+2,3*(N-1)+2] += kbot
        B[3*(N-1)+2,3*(N-1)+2] += cbot
    
    return M,A,B,K,r_mean
      
def get_leg_matrices(ms,fairlead_id,omegas,S_zeta,moor_dict,r_dynamic):
    """
    Evaluates mooring leg dynamic equation of motion matrices. A mooring leg here is defined as a serial assembly of moorpy Line objects.

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
    EA_segs = np.zeros(n_nodes-1)     # TODO: EA_seg = np.zeros(n_nodes-1)
    n_dofs = 3*n_nodes # number of degrees of freedom is equal to total number of nodes minus the number of shared nodes x3
    M = np.zeros([n_dofs,n_dofs], dtype='float')
    A = np.zeros([n_dofs,n_dofs], dtype='float')
    B = np.zeros([n_dofs,n_dofs], dtype='float')
    K = np.zeros([n_dofs,n_dofs], dtype='float')
    r_nodes = np.zeros([n_nodes,3], dtype='float')
    n = 0
    r_dynamic = np.ones((len(omegas),n_nodes,3),dtype='float')*r_dynamic
    for Line in lines:
        LineType = ms.lineTypes[Line.type]
        # Filling matrices for line (dof n to dof 3xline_nodes+n)
        Mn,An,Bn,Kn,rn = get_line_matrices(Line,LineType,omegas,S_zeta,r_dynamic,depth,kbot,cbot,seabed_tol=1e-4)
        M[n:3*Line.nNodes+n,n:3*Line.nNodes+n] += Mn
        A[n:3*Line.nNodes+n,n:3*Line.nNodes+n] += An
        B[n:3*Line.nNodes+n,n:3*Line.nNodes+n] += Bn
        K[n:3*Line.nNodes+n,n:3*Line.nNodes+n] += Kn
        
        M[3*Line.nNodes - 3:3*Line.nNodes, 3*Line.nNodes - 3:3*Line.nNodes] += ms.pointList[Line.attached[-1]-1].m*np.eye(3)
        n_node0 = int(n/3)
        r_nodes[n_node0:Line.nNodes+n_node0,:] = rn
        EA_segs[n_node0:Line.nNodes-1] = LineType.EA 
        # TODO: add added mass and drag coefficient (add Cd and Ca to point object attribute)
        
        n += 3*Line.nNodes - 3 # next line starting node add the number of dofs of the current line minus 3 to get the last shared node
    
    return M,A,B,K,n_dofs,r_nodes,EA_segs

def get_dynamic_tension(ms,fairlead_id,moor_dict,omegas,S_zeta,RAOs,tol = 0.01,iters=100, w = 0.8, no_fail=True, finite_difference=False):
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
    M,A,B,K,n_dofs,r_static,EA_e = get_leg_matrices(ms,fairlead_id,omegas,S_zeta,moor_dict,1.)
    X = np.zeros((len(omegas),n_dofs),dtype = 'complex')
    r_dynamic = np.zeros(((len(omegas),int(n_dofs/3),3)),dtype = 'complex')
    S_Xd = np.zeros((len(omegas),n_dofs),dtype = 'float')
    sigma_Xd = np.zeros(n_dofs,dtype = 'float')
    sigma_Xd0 = np.zeros(n_dofs,dtype = 'float')
    X[:,:3] = RAO_fl

    # solving dynamics
    start = datetime.now()
    for ni in range(iters):
        H = - omegas[:,np.newaxis,np.newaxis]**2*(M+A)[np.newaxis,:,:]\
            + 1j*omegas[:,np.newaxis,np.newaxis]*B[np.newaxis,:,:]\
            + K[np.newaxis,:,:]\
        
        F = np.einsum('nij,njk->ni',-H[:,3:-3,:3],RAO_fl[:,:,np.newaxis])

        X[:,3:-3] = solve(H[:,3:-3,3:-3],F)
        S_Xd[:] = np.abs(1j*omegas[:,np.newaxis]*X)**2*S_zeta[:,np.newaxis]
        sigma_Xd[:] = np.sqrt(np.trapz(S_Xd,omegas,axis=0)) 
        r_dynamic[:] = X.reshape(X.shape[0],int(X.shape[1]/3),3)
        if (np.abs(sigma_Xd-sigma_Xd0) <= tol*np.abs(sigma_Xd0)).all():
            break
        else:
            sigma_Xd0[:] = w * sigma_Xd + (1.-w) * sigma_Xd0
            _,_,B[:],_,_,_,_ = get_leg_matrices(ms,fairlead_id,omegas,S_zeta,moor_dict,r_dynamic)
    print(f'Finished {ni} dynamic tension iterations in {datetime.now()-start} seconds (w = {w}).')

    # evaluate tension
    dw = np.diff(omegas,
             prepend= omegas[0] - (omegas[1]-omegas[0]),
             append= omegas[-1] + (omegas[-1]-omegas[-2]))
    dw = (dw[1:]+dw[:-1])/2
    wave_amps = np.sqrt(S_zeta*dw) #evaluate wave amplitudes of harmonic components from wave spectrum

    r_dynamic *= wave_amps[:,np.newaxis,np.newaxis]
    r_total = r_static[np.newaxis,:,:] + r_dynamic
    dr_static = r_static[:-1] - r_static[1:]
    dr_dynamic = r_dynamic[:,:-1,:] - r_dynamic[:,1:,:]
    tangents = dr_static/la.norm(r_static[:-1] - r_static[1:], axis=-1)[:,np.newaxis]
    L_static = la.norm(dr_static, axis=-1)
    dL_dynamic = np.einsum('mni,ni->mn', dr_dynamic, tangents)
    eps_e = np.abs(dL_dynamic)/L_static

    T_e = EA_e[np.newaxis,:] * eps_e
    S_T = T_e**2/dw[:,np.newaxis]
    sigma_T = np.sqrt(np.trapz(S_T,omegas,axis=0))

    print(np.abs(eps_e).max())
    dr = np.diff(r_static,axis=0)
    ds = la.norm(dr,axis=1)
    s = np.zeros(len(sigma_T))
    s = np.cumsum(ds)
    
    return sigma_T,S_T,s,r_static,r_dynamic,r_total,X

def get_leg_modes(ms,fairlead_id,moor_dict):
    fairlead = ms.pointList[fairlead_id - 1]
    M,A,B,K,n_dofs,r_nodes,_ = get_leg_matrices(ms,fairlead.number,np.ones(1),np.ones(1),moor_dict,1.)

    eigvals,eigvecs = la.eig(K[3:-3,3:-3],M[3:-3,3:-3]+A[3:-3,3:-3])
    stable_eigvals = eigvals[np.real(eigvals)>0]
    stable_eigvecs = eigvecs[:,np.real(eigvals)>0]
    
    idx = stable_eigvals.argsort()[::-1]   
    stable_eigvals = stable_eigvals[idx]
    stable_eigvecs = stable_eigvecs[:,idx]
   
    freqs = np.sqrt(np.real(stable_eigvals))/2/np.pi
    mode_shapes = np.zeros(stable_eigvecs.shape,dtype='float')
    
    for i in range(stable_eigvecs.shape[1]):
        mode_shapes[:,i] = r_nodes[1:-1].flatten('C') + stable_eigvecs[:,i]

    freqs = np.flip(freqs)
    mode_shapes = np.flip(mode_shapes,axis=1)    

    return freqs,mode_shapes,r_nodes,M,A,K

def animate_line_motion(omega,omegas,r_nodes,X_nodes,Amp=1.,t_sim=60,dt=0.1,fps=20):
    x_0 = r_nodes[:,0]
    y_0 = r_nodes[:,1]
    z_0 = r_nodes[:,2]
    time = np.arange(0, t_sim+dt, dt)
    X_func = interp1d(omegas,X_nodes,axis=0)
    X_omega = X_func(omega)
    fig,ax = plt.subplots(1,1,subplot_kw={"projection": "3d"})
    line = ax.plot(x_0,y_0,z_0,color='black',marker='o', markersize = 3)[0]
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    # r_0 = np.sqrt(x_0**2 + y_0**2)
    # ax.set_xlim(-r_0.max(),r_0.max())
    # ax.set_ylim(-r_0.max(),r_0.max())
    # ax.set_zlim(z_0.min(),z_0.max())
    ax.set_aspect('equal')

    fig.suptitle(f'Wave period {2*np.pi/omega:04.1f} s, Wave ampliude {Amp:04.1f} m')
    
    def update_frame(frn,time,line,Amp,X_omega,fig,ax):
        t = time[frn]
        x = x_0 + [np.real(Amp*np.abs(X_omega[i])*np.exp(1j*(omega*t+np.angle(X_omega[i])))) for i in range(0,len(X_omega),3)]
        y = y_0 + [np.real(Amp*np.abs(X_omega[i])*np.exp(1j*(omega*t+np.angle(X_omega[i])))) for i in range(1,len(X_omega),3)]
        z = z_0 + [np.real(Amp*np.abs(X_omega[i])*np.exp(1j*(omega*t+np.angle(X_omega[i])))) for i in range(2,len(X_omega),3)]
        line.set_data_3d(x,y,z)
        ax.set_title(f'Time = {t:03.1f} s')
        return line
        
    line_animation = animation.FuncAnimation(fig, update_frame, repeat=True, fargs=(time,line,Amp,X_omega,fig,ax),
                                             frames=len(time), interval=dt/fps*1e3)
    
    return line_animation

def plot_mode_shapes(mode_idxs,freqs,mode_shapes,r_nodes,amp_factor = 10.,figsize=5,cols=3,adj_view = True):
    from collections.abc import Iterable
    if not isinstance(mode_idxs,Iterable):
        mode_idxs = [mode_idxs]
    rows = len(mode_idxs)//cols + bool(len(mode_idxs)%cols)
    fig,ax = plt.subplots(rows,cols,subplot_kw={"projection": "3d"},figsize=(figsize*cols,figsize*rows))
    
    i = 0 
    for axes in ax:
        if not isinstance(axes,Iterable):
            axes = [axes]

        for axis in axes:
            if i >= len(mode_idxs):
                break
            idx = mode_idxs[i]
            r = r_nodes.copy()
            r[1:-1] = mode_shapes[:,idx].reshape([int(len(mode_shapes[:,idx])/3),3])
            r = (r-r_nodes)*amp_factor
            x = r[:,0]
            y = r[:,1]
            z = r[:,2]
            
            x_0 = r_nodes[:,0]
            y_0 = r_nodes[:,1]
            z_0 = r_nodes[:,2]
                
            axis.plot(x_0,y_0,z_0,'-ko',label='initial')
            axis.plot(x+x_0,y+y_0,z+z_0,'--ro',label='mode shape')
            
            # R_0 = np.sqrt(x_0**2 + y_0**2)
            if adj_view:
                # h_min = np.min((x_0,y_0))
                # h_max = np.max((x_0,y_0))
                # axis.set_xlim(h_min,h_max)
                # axis.set_ylim(h_min,h_max)
                # axis.set_zlim(z_0.min(),z_0.max())
                sigma_x = x.std() 
                sigma_y = y.std()
                sigma_z = z.std()
                azim = np.arctan2(sigma_x,sigma_y)*180/np.pi
                elev = np.arctan2(np.hypot(sigma_x,sigma_y),sigma_z)*180/np.pi
                axis.view_init(elev=elev,azim=azim)
            axis.set_aspect('equal', 'box')
            axis.set_xlabel('X (m)')
            axis.set_ylabel('Y (m)')
            axis.set_zlabel('Z (m)')
            axis.set_title(f'f = {freqs[idx]:.3f} Hz\nT = {1/freqs[idx]:.3f} s')

            i+=1

    fig.tight_layout()
    return ax


#%%

# def get_dynamic_tension(ms,fairlead_id,moor_dict,omegas,S_zeta,RAOs,tol = 0.01,iters=100, w = 0.8, no_fail=True, finite_difference=False):
#     """Evaluates dynamic tension standard deviations along a mooring leg.

#     Parameters
#     ----------
#     ms : moorpy System object
#         moorpy mooring System object at mean position.
#     fairlead_id : int
#         Index of the fairlead Point object in the moorpy Body object at which the mooring leg starts.        
#     moor_dict : dictionary
#         dictionary with the keys specified in the spread_mooring function in the mooring_configs module.
#     omegas : array
#         Frequencies at which to solve line dynamics in rad/s.
#     S_zeta : array
#         Wave elevation spectral density at specified frequencies in m^2.s/rad
#     RAOs : 2d numpy array
#         First order Response Ampltiude operators for 6 dofs (columns) at the specified frequencies (rows)
#     tol : float, optional
#         Relative tolerence for iteration convergence, by default 0.01
#     iters : int, optional
#         Maximum number of iterations, by default 100

#     Returns
#     -------
#     sigma_T : array
#         Dynamic tension standard deviation at nodes.
#     S_T : 2d numpy array
#         Dynamic tension spectra at nodes
#     s : array
#         Along line locations of nodes corresponding to the calculated tensions.
#     r_nodes : 2d numpy array array
#         Node coordinates.
#     X : array
#         Node dofs motion responses.
    
#     """
#     body = ms.bodyList[0]
#     # evaluate top end motion
#     fairlead = ms.pointList[fairlead_id - 1]
#     r_fl = fairlead.r - body.r6[:3] #floater cg to fairlead vector
#     RAO_fl = RAOs[:,:3] + np.cross(RAOs[:,3:],r_fl[np.newaxis,:],axisa=-1,axisb=-1)

#     # get quasi-static configuration
#     offset = 2*np.sqrt(np.trapz(np.abs(RAO_fl)**2*S_zeta[:,np.newaxis],omegas,axis=0))
#     ms_qs = copy.deepcopy(ms)
#     ms_qs.bodyList[0].type = -1
#     ms_qs.bodyList[0].r6[:3] += offset
#     ms_qs.initialize()
#     conv = ms_qs.solveEquilibrium(tol=tol, maxIter=iters, no_fail=no_fail, finite_difference=finite_difference)

#     # intialize iteration matrices    
#     M,A,B,K,n_dofs,r_static,EA_e = get_leg_matrices(ms,ms_qs,fairlead.number,moor_dict,1.)
#     X = np.zeros((len(omegas),n_dofs),dtype = 'complex')
#     S_Xd = np.zeros((len(omegas),n_dofs),dtype = 'float')
#     sigma_Xd = np.zeros(n_dofs,dtype = 'float')
#     sigma_Xd0 = np.zeros(n_dofs,dtype = 'float')
#     X[:,:3] = RAO_fl

#     # Rayleigh damping
#     rayleigh_flag = 0.
#     omega_1 = 2*np.pi/20
#     omega_2 = 2*np.pi/4
#     xi_1 = 1.
#     xi_2 = 1.
#     alpha = 2*omega_1*omega_2*(xi_1*omega_2-xi_2*omega_1)/(omega_2**2-omega_1**2) 
#     beta = 2*(xi_2*omega_2-xi_1*omega_1)/(omega_2**2-omega_1**2) 
    
#     # get geometric stiffness matrix TODO
#     print(offset)
#     K_gs = _get_linearized_geometric_stiffness(ms,offset,fairlead_id,
#                                          tol=0.01, maxIter=500, no_fail=True, finite_difference=False)
#     K += K_gs*0.
#     # solving dynamics
#     start = datetime.now()
#     for ni in range(iters):
#         H = - omegas[:,np.newaxis,np.newaxis]**2*(M+A)[np.newaxis,:,:]\
#             + 1j*omegas[:,np.newaxis,np.newaxis]*(B)[np.newaxis,:,:]\
#             + K[np.newaxis,:,:]\
#             + 1j*omegas[:,np.newaxis,np.newaxis]*(alpha*(M+A)[np.newaxis,:,:]+beta*K[np.newaxis,:,:])*rayleigh_flag 
        
#         F = np.einsum('nij,njk->ni',-H[:,3:-3,:3],RAO_fl[:,:,np.newaxis])

#         X[:,3:-3] = solve(H[:,3:-3,3:-3],F)
#         # X[:,3:-3] = la.solve(H[:,3:-3,3:-3],F,assume_a='sym',check_finite=False)
#         S_Xd[:] = np.abs(1j*omegas[:,np.newaxis]*X)**2*S_zeta[:,np.newaxis]
#         sigma_Xd[:] = np.sqrt(np.trapz(S_Xd,omegas,axis=0)) 

#         if (np.abs(sigma_Xd-sigma_Xd0) <= tol*np.abs(sigma_Xd0)).all():
#             break
#         else:
#             sigma_Xd0[:] = w * sigma_Xd + (1.-w) * sigma_Xd0
#             _,_,B[:],_,_,_,_ = get_leg_matrices(ms,ms_qs,fairlead.number,moor_dict,sigma_Xd0)
#     print(f'Finished {ni} dynamic tension iterations in {datetime.now()-start} seconds (w = {w}).')

#     # evaluate tension
#     dw = np.diff(omegas,
#              prepend= omegas[0] - (omegas[1]-omegas[0]),
#              append= omegas[-1] + (omegas[-1]-omegas[-2]))
#     dw = (dw[1:]+dw[:-1])/2
#     wave_amps = np.sqrt(S_zeta*dw) #evaluate wave amplitudes of harmonic components from wave spectrum

#     r_dynamic = wave_amps[:,np.newaxis,np.newaxis]*X.reshape(X.shape[0],int(X.shape[1]/3),3)
#     r_total = r_static[np.newaxis,:,:] + r_dynamic

#     L_static = la.norm(r_static[:-1] - r_static[1:], axis=-1)
#     L_total = (la.norm(r_total[:,:-1,:] - r_total[:,1:,:], axis=-1))

#     eps_e = (L_total-L_static[np.newaxis,:])/L_static[np.newaxis,:]
#     T_e = EA_e[np.newaxis,:] * eps_e
#     S_T = T_e**2/dw[:,np.newaxis]

#     #TODO
#     # X_nodes = X.reshape(X.shape[0],int(X.shape[1]/3),3)
#     # dX_e = X_nodes[:,:-1,:] - X_nodes[:,1:,:] # change in length
#     # dr_e = r_nodes[:-1] - r_nodes[1:] # change in length
#     # dL_e = la.norm(dr_e, axis=-1)
#     # t_e = dr_e/dL_e[:,np.newaxis]
#     # eps_e = la.norm(np.einsum('nij,ij->ni',dX_e,t_e))/dL_e[np.newaxis,:] #dot product between displacement change and tangential unit vec
#     # T_e = EA_segs[np.newaxis,:] * eps_e 
#     # S_T = np.abs(T_e)**2*S_zeta[:,np.newaxis]

#     # evaluating motion to tension transfer function
#     # H_zF = np.zeros([len(omegas),n_dofs],dtype='complex')
#     # H_zT = np.zeros([len(omegas),int(n_dofs/3)],dtype='complex')

#     # start = datetime.now()
#     # for nw in range(len(omegas)):
#     #     for n in range(int(n_dofs/3 - 1)):
#     #         H_zF[nw,3*n:3*n+3] = np.matmul(K[3*n:3*n+3,3*n+3:3*n+6],(X[nw,3*n+3:3*n+6] - X[nw,3*n:3*n+3]))
#     #         H_zT[nw,n] = la.norm(H_zF[nw,3*n:3*n+3])
    
#     #     H_zF[nw,-3:] = np.matmul(K[-3:,-3:],(X[nw,-6:-3] - X[nw,-3:]))
#     #     H_zT[nw,-1] = la.norm(H_zF[nw,-3:])

    
#     # S_T = np.abs(H_zT[:,:-1])**2*S_zeta[:,np.newaxis]
#     # sigma_T = np.sqrt(np.trapz(S_T,omegas,axis=0))
#     # print(f'Finished dynamic tension calculations in {datetime.now()-start} seconds.')

#     print(np.abs(eps_e).max())
#     sigma_T = np.sqrt(np.trapz(S_T,omegas,axis=0))
#     dr = np.diff(r_static,axis=0)
#     ds = la.norm(dr,axis=1)
#     s = np.zeros(len(sigma_T))
#     s = np.cumsum(ds)
    
#     return sigma_T,S_T,s,r_static,r_dynamic,r_total,X

# def get_line_matrices(Line, LineType, QSLine, sigma_u, depth, kbot, cbot, seabed_tol = 1e-4): #TODO: Make QSLINE optional?
#     """
#     Evaluates mooring line dynamic equation of motion matrices.

#     Parameters
#     ----------
#     Line : moorpy.Line object (modified package)
#     LineType : moorpy.LineType object (modified package)
#     sigma_u : 1D numpy array
#         Standard deviation velocity vector of line nodes.
#     depth : float
#         Seabed depth.
#     kbot : float
#         Seabed vertical stiffness coefficient.
#     cbot : float
#         Seabed vertical damping coefficient.

#     Returns
#     -------
#     M : 2D numpy array
#         Line mass matrix.
#     A : 2D numpy array
#         Line added mass matrix.
#     B : 2D numpy array
#         Line linearized viscous damping matrix.
#     K : 2D numpy array
#         Line stiffness matrix.
#     r_nodes : 2D numpy array
#         Locations of nodes (each row represents the coords of a node)
#     """
    
#     n_nodes = Line.nNodes
#     n_lines = n_nodes - 1 # NOTE: I am using leg->line->segment instead of line->segment->element to be consistent with moorpy's notation
#     mden = LineType.mlin # line mass density function
#     deq = LineType.d # line volume equivalent diameter
#     Le = Line.L/n_lines # line segment (element) length
#     me =  mden*Le # line segment (element) mass
#     EA = LineType.EA # extensional stiffness
#     Can = LineType.Can # normal added mass coeff
#     Cat = LineType.Cat # tangential added mass coeff
#     Cdn = LineType.Cdn # normal drag coeff
#     Cdt = LineType.Cdt # tangential drag coeff


#     X_nodes,Y_nodes,Z_nodes,T_nodes = Line.getCoordinate(np.linspace(0,1,n_nodes)*Line.L) # coordinates of line nodes and tension values
#     r_nodes = np.vstack((X_nodes,Y_nodes,Z_nodes)).T # coordinates of line nodes

#     slopes = np.gradient(r_nodes,Le,axis=0,edge_order=2)
#     theta = np.arctan2(slopes[:,2],np.sqrt(slopes[:,0]**2+slopes[:,1]**2))
#     s = np.linspace(0,1,n_nodes)*Line.L
#     dtheta_ds = np.gradient(theta,s,edge_order=2)

#     X_qs,Y_qs,Z_qs,T_qs = QSLine.getCoordinate(np.linspace(0,1,n_nodes)*QSLine.L) # coordinates of line nodes and tension values
#     r_qs = np.vstack((X_qs,Y_qs,Z_qs)).T # coordinates of line nodes
#     dr_qs = r_nodes - r_qs

#     M = np.zeros([3*n_nodes, 3*n_nodes], dtype='float') # mass matrix
#     A = np.zeros([3*n_nodes, 3*n_nodes], dtype='float') # added mass matrix
#     B = np.zeros([3*n_nodes, 3*n_nodes], dtype='float') # linearized viscous damping matrix
#     K = np.zeros([3*n_nodes, 3*n_nodes], dtype='float') # stiffness matrix
    
#     kg_flag = 1. # NOTE: added to test effect of geometric stiffness
#     kt_flag = 0.
#     ks_flag = 0.

#     # Node 1 (fairlead)
#     M[0:3,0:3] += me/2*np.eye(3) # element 1 mass contribution
    
#     L_e2 = la.norm(r_nodes[1] - r_nodes[0]) # element 1 length
#     t_e2 = (r_nodes[1] - r_nodes[0])/L_e2 # tangential unit vector

#     dt_e2 = np.abs((r_qs[1] - r_qs[0])/L_e2 - t_e2) # tangential unit vector
#     t_e2 += dt_e2 * kt_flag

#     sigma_ut2 = np.dot(sigma_u[0:3],t_e2) # standard deviation of tangential velocity
#     sigma_un2 = np.sqrt(np.abs(la.norm(sigma_u[0:3])**2 - sigma_ut2**2)) # standard deviation of normal velocity
    
#     Rt_e2 = np.outer(t_e2,t_e2) # local tangential to global components transformation matrix
#     Rn_e2 = np.eye(3) - Rt_e2 # local normal to global components transformation matrix
    
#     A_e2 = 1025*np.pi/4*deq**2*L_e2/2*(Can*Rn_e2 + Cat*Rt_e2) # element 1 added mass contribution
#     B_e2 = 0.5*1025*deq*L_e2/2*np.sqrt(8/np.pi)*(Cdn*sigma_un2*Rn_e2 + 
#                                                  Cdt*sigma_ut2*Rt_e2) # element 1 damping contribution 
#     K_e2 = EA/L_e2*Rt_e2 + T_nodes[0]/L_e2*Rn_e2 * kg_flag  # element 1 stiffness (axial + geometric)
    
#     A[0:3,0:3] += A_e2 
#     B[0:3,0:3] += B_e2
#     K[0:3,0:3] += K_e2 - EA*dtheta_ds[0]*Rn_e2 * ks_flag 
#     K[0:3,3:6] += -K_e2
    
#     if np.isclose(r_nodes[0,2],-depth,seabed_tol):
#         K[2,2] += kbot
#         B[2,2] += cbot 
    
#     # Internal nodes loop (each internal node has contributions from two elements n-1/2 and n+1/2)
#     for n in range(1, n_nodes-1):
        
#         M[3*n:3*n+3,3*n:3*n+3] += me*np.eye(3) # mass contribution from adjacent elements
        
#         ## element n-1/2 contributions
#         L_e1 = la.norm(r_nodes[n] - r_nodes[n-1]) # element n-1/2 length
#         t_e1 = (r_nodes[n] - r_nodes[n-1])/L_e1 # element n-1/2 tangential unit vector

#         dt_e1 = np.abs((r_qs[n] - r_qs[n-1])/L_e1 - t_e1) # tangential unit vector
#         t_e1 += dt_e1 * kt_flag

#         sigma_ut1 = np.dot(sigma_u[3*n:3*n+3],t_e1) # standard deviation of tangential velocity
#         sigma_un1 = np.sqrt(np.abs(la.norm(sigma_u[3*n:3*n+3])**2 - sigma_ut1**2)) # standard deviation of normal velocity
#         Rt_e1 = np.outer(t_e1,t_e1)
#         Rn_e1 = np.eye(3) - Rt_e1
        
#         A_e1 = 1025*np.pi/4*deq**2*L_e1/2*(Can*Rn_e1 + Cat*Rt_e1) # element n-1/2 added mass contribution
        
#         B_e1 = 0.5*1025*deq*L_e1/2*np.sqrt(8/np.pi)*(Cdn*sigma_un1*Rn_e1 + 
#                                                      Cdt*sigma_ut1*Rt_e1) # element n-1/2 damping contribution
        
#         K_e1 = EA/L_e1*Rt_e1 + T_nodes[n]/L_e1*Rn_e1 * kg_flag 
       
#         ## element n+1/2 contributions
#         L_e2 = la.norm(r_nodes[n+1] - r_nodes[n])
#         t_e2 = (r_nodes[n+1] - r_nodes[n])/L_e2

#         dt_e2 = np.abs((r_qs[n+1] - r_qs[n])/L_e2 - t_e2) # tangential unit vector
#         t_e2 += dt_e2 * kt_flag
        
#         sigma_ut2 = np.dot(sigma_u[3*n:3*n+3],t_e2)
#         sigma_un2 = np.sqrt(np.abs(la.norm(sigma_u[3*n:3*n+3])**2 - sigma_ut2**2))
#         Rt_e2 = np.outer(t_e2,t_e2) # local tangential to global components transformation matrix
#         Rn_e2 = np.eye(3) - Rt_e2 # local normal to global components transformation matrix
#         A_e2 = 1025*np.pi/4*deq**2*L_e2/2*(Can*Rn_e2 + Cat*Rt_e2) # element n+1/2 added mass contribution
        
#         B_e2 = 0.5*1025*deq*L_e2/2*np.sqrt(8/np.pi)*(Cdn*sigma_un2*Rn_e2 + 
#                                                      Cdt*sigma_ut2*Rt_e2) # element n-1/2 damping contribution
        
#         K_e2 = EA/L_e2*Rt_e2 + T_nodes[n]/L_e2*Rn_e2 * kg_flag
        
#         ## fill line matrices
#         A[3*n:3*n+3,3*n:3*n+3] += A_e1 + A_e2 # added mass
#         B[3*n:3*n+3,3*n:3*n+3] += B_e1 + B_e2 # damping
#         K[3*n:3*n+3,3*n:3*n+3] += K_e1 + K_e2 - EA*dtheta_ds[n]*Rn_e1 * ks_flag
#         K[3*n:3*n+3,3*n-3:3*n] += -K_e1
#         K[3*n:3*n+3,3*n+3:3*n+6] += -K_e2
        
#         ## add seabed contribution to node n
#         if np.isclose(r_nodes[n,2],-depth,seabed_tol):
#             K[3*n+2,3*n+2] += kbot
#             B[3*n+2,3*n+2] += cbot 
    
#      # Node N (anchor)
#     M[3*(n_nodes-1):3*(n_nodes-1)+3,3*(n_nodes-1):3*(n_nodes-1)+3] += me/2*np.eye(3) # element N-1 mass contribution
    
#     L_e1 = la.norm(r_nodes[n_nodes-1] - r_nodes[n_nodes-2]) # element N-1 length
#     t_e1 = 1/L_e2*(r_nodes[n_nodes-1] - r_nodes[n_nodes-2]) # tangential unit vector

#     dt_e1 = np.abs((r_qs[n_nodes-1] - r_qs[n_nodes-2])/L_e1 - t_e1) # tangential unit vector
#     t_e1 += dt_e1 * kt_flag

#     sigma_ut1 = np.dot(sigma_u[3*(n_nodes-1):3*(n_nodes-1)+3],t_e1) # standard deviation of tangential velocity 
#     sigma_un1 = np.sqrt(np.abs(la.norm(sigma_u[3*(n_nodes-1):3*(n_nodes-1)+3])**2 - sigma_ut1**2)) # standard deviation of normal velocity
#     Rt_e1 = np.outer(t_e2,t_e2) # local tangential to global components transformation matrix
#     Rn_e1 = np.eye(3) - Rt_e1 # local normal to global components transformation matrix
#     A_e1 = 1025*np.pi/4*deq**2*L_e1/2*(Can*Rn_e1 + Cat*Rt_e1) # element N-1 added mass contribution
    
#     B_e1 = 0.5*1025*deq*L_e1/2*np.sqrt(8/np.pi)*(Cdn*sigma_un1*Rn_e1 + 
#                                                  Cdt*sigma_ut1*Rt_e1) # element N-1 damping contribution
    
#     K_e1 = EA/L_e1*Rt_e1 + T_nodes[-1]/L_e1*Rn_e1 * kg_flag 
    
#     A[3*(n_nodes-1):3*(n_nodes-1)+3,3*(n_nodes-1):3*(n_nodes-1)+3] += A_e1 # added mass
#     B[3*(n_nodes-1):3*(n_nodes-1)+3,3*(n_nodes-1):3*(n_nodes-1)+3] += B_e1 # linearized viscous damping
#     K[3*(n_nodes-1):3*(n_nodes-1)+3,3*(n_nodes-1):3*(n_nodes-1)+3] += K_e1 - EA*dtheta_ds[-1]*Rn_e2 * ks_flag # stiffness matrix
#     K[3*(n_nodes-1):3*(n_nodes-1)+3,3*(n_nodes-1)-3:3*(n_nodes-1)] += -K_e1 # stiffness matrix
    
#     ## add seabed contribution to node N
#     if np.isclose(r_nodes[n_nodes-1,2],-depth,seabed_tol):
#         K[3*(n_nodes-1)+2,3*(n_nodes-1)+2] += kbot
#         B[3*(n_nodes-1)+2,3*(n_nodes-1)+2] += cbot
    
#     # B += alpha*(M+A) + beta*K # Rayleigh damping

#     return M,A,B,K,r_nodes

# def _get_linearized_geometric_stiffness(ms_mean,offset,fairlead_id,
#                                         tol=0.01, maxIter=500, no_fail=True, finite_difference=False):

#     line_ids,point_ids = get_mooring_leg(ms_mean, fairlead_id) #NOTE: to be removed
#     mean_lines = [ms_mean.lineList[line_id - 1] for line_id in line_ids] #NOTE: to be removed
#     n_nodes = (np.sum([line.nNodes for line in mean_lines]) - (len(mean_lines)-1)) #NOTE: to be removed
#     n_dofs = 3*n_nodes

#     K_gs = np.zeros([n_dofs,n_dofs],dtype='float')
#     for i in range(len(offset)):
#         if not np.isclose(np.abs(offset[i]),0.):
#             ms_offset = copy.deepcopy(ms_mean)
#             ms_offset.bodyList[0].type = -1
#             ms_offset.bodyList[0].r6[i] += offset[i]
#             ms_offset.initialize()
#             conv = ms_offset.solveEquilibrium(tol=tol, maxIter=maxIter, no_fail=no_fail, finite_difference=finite_difference)
#             offset_lines = [ms_offset.lineList[line_id - 1] for line_id in line_ids]

#             n = 0
#             for MeanLine,OffsetLine in zip(mean_lines,offset_lines):
#                 Le = MeanLine.L/(MeanLine.nNodes-1)

#                 X_mean,Y_mean,Z_mean,T_mean = MeanLine.getCoordinate(np.linspace(0,1,MeanLine.nNodes)*MeanLine.L) # coordinates of line nodes and tension values
#                 r_mean = np.vstack((X_mean,Y_mean,Z_mean)).T # coordinates of line nodes

#                 X_offset,Y_offset,Z_offset,T_offset = OffsetLine.getCoordinate(np.linspace(0,1,MeanLine.nNodes)*OffsetLine.L) # coordinates of line nodes and tension values
#                 r_offset = np.vstack((X_offset,Y_offset,Z_offset)).T # coordinates of line nodes

#                 mean_slopes = np.gradient(r_mean,Le,axis=0,edge_order=2)
#                 offset_slopes = np.gradient(r_offset,Le,axis=0,edge_order=2)

#                 F_mean = T_mean[:,np.newaxis]*mean_slopes
#                 F_offset = T_offset[:,np.newaxis] * offset_slopes
                
#                 # dr = (r_offset - r_mean).flatten()
#                 dF = (F_offset-F_mean).flatten()
                
#                 K_gs[n:3*MeanLine.nNodes+n,i] = np.abs(dF/offset[i])

#                 n += 3*MeanLine.nNodes - 3
            
#     K_gs[:3,3:] = K_gs[3:,:3].T
#     return K_gs

# def get_leg_matrices(ms,ms_qs,fairlead_id,moor_dict, sigma_u):
#     """Evaluates mooring leg dynamic equation of motion matrices. A mooring leg here is defined as a serial assembly of moorpy Line objects.

#     Parameters
#     ----------
#     ms : moorpy System object
#         moorpy System object at the mean position.
#     fairlead_id : int
#         Index of the fairlead Point object in the moorpy Body object at which the mooring leg starts.
#     moor_dict : dictionary
#         dictionary with the keys specified in the spread_mooring function in the mooring_configs module.
#     sigma_u : float or array
#         Nodes DOFs velocity standard deviation.

#     Returns
#     -------
#     M : 2D numpy array
#         Leg mass matrix.
#     A : 2D numpy array
#         Leg added mass matrix.
#     B : 2D numpy array
#         Leg linearized viscous damping matrix.
#     K : 2D numpy array
#         Leg stiffness matrix.
#     n_dofs: int
#         Number of degrees of freedom.
#     r_nodes : 2D numpy array
#         Locations of nodes (each row represents the coords of a node)
#     """
    
#     line_ids,point_ids = get_mooring_leg(ms, fairlead_id)
#     lines = [ms.lineList[line_id - 1] for line_id in line_ids]
#     qs_lines = [ms_qs.lineList[line_id - 1] for line_id in line_ids]
#     depth = ms.depth
#     kbot = moor_dict['kbot']
#     cbot = moor_dict['cbot']
    
#     n_nodes = (np.sum([line.nNodes for line in lines]) - (len(lines)-1))
#     EA_segs = np.zeros(n_nodes-1)     # TODO: EA_seg = np.zeros(n_nodes-1)
#     n_dofs = 3*n_nodes # number of degrees of freedom is equal to total number of nodes minus the number of shared nodes x3
#     M = np.zeros([n_dofs,n_dofs], dtype='float')
#     A = np.zeros([n_dofs,n_dofs], dtype='float')
#     B = np.zeros([n_dofs,n_dofs], dtype='float')
#     K = np.zeros([n_dofs,n_dofs], dtype='float')
#     r_nodes = np.zeros([n_nodes,3], dtype='float')
#     n = 0
#     sigma_u = np.ones(n_dofs,dtype='float')*sigma_u
#     for Line,QSLine in zip(lines,qs_lines):
#         LineType = ms.lineTypes[Line.type]
#         # Filling matrices for line (dof n to dof 3xline_nodes+n)
#         Mn,An,Bn,Kn,rn = get_line_matrices(Line, LineType, QSLine, sigma_u[n:3*Line.nNodes+n], depth, kbot, cbot)
#         M[n:3*Line.nNodes+n,n:3*Line.nNodes+n] += Mn
#         A[n:3*Line.nNodes+n,n:3*Line.nNodes+n] += An
#         B[n:3*Line.nNodes+n,n:3*Line.nNodes+n] += Bn
#         K[n:3*Line.nNodes+n,n:3*Line.nNodes+n] += Kn
        
#         M[3*Line.nNodes - 3:3*Line.nNodes, 3*Line.nNodes - 3:3*Line.nNodes] += ms.pointList[Line.attached[-1]-1].m*np.eye(3)
#         n_node0 = int(n/3)
#         r_nodes[n_node0:Line.nNodes+n_node0,:] = rn
#         EA_segs[n_node0:Line.nNodes-1] = LineType.EA # TODO: EA_seg[:] = Line.EA
#         # TODO: add added mass and drag coefficient (add Cd and Ca to point object attribute)
        
#         n += 3*Line.nNodes - 3 # next line starting node add the number of dofs of the current line minus 3 to get the last shared node
    
#     return M,A,B,K,n_dofs,r_nodes,EA_segs # TODO: add EA_seg array as out


# def get_leg_modes(ms,ms_qs,fairlead_id,moor_dict):
#     fairlead = ms.pointList[fairlead_id - 1]
#     M,A,B,K,n_dofs,r_nodes,_ = get_leg_matrices(ms,ms_qs,fairlead.number,moor_dict,0.)

#     eigvals,eigvecs = la.eig(K[3:-3,3:-3],M[3:-3,3:-3]+A[3:-3,3:-3])
#     stable_eigvals = eigvals[np.real(eigvals)>0]
#     stable_eigvecs = eigvecs[:,np.real(eigvals)>0]
    
#     idx = stable_eigvals.argsort()[::-1]   
#     stable_eigvals = stable_eigvals[idx]
#     stable_eigvecs = stable_eigvecs[:,idx]
   
#     freqs = np.sqrt(np.real(stable_eigvals))/2/np.pi
#     mode_shapes = np.zeros(stable_eigvecs.shape,dtype='float')
    
#     for i in range(stable_eigvecs.shape[1]):
#         mode_shapes[:,i] = r_nodes[1:-1].flatten('C') + stable_eigvecs[:,i]

#     freqs = np.flip(freqs)
#     mode_shapes = np.flip(mode_shapes,axis=1)    

#     return freqs,mode_shapes,r_nodes,M,A,K
