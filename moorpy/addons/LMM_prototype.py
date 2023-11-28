import numpy as np
import scipy.linalg as la

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

def get_line_matrices2(Line,LineType,omegas,S_zeta,v_nodes,depth,kbot,cbot,seabed_tol=1e-4):
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

    ut_e1 = np.einsum('ij,j->i',v_nodes[:,0,:],t_e1) # tangential velocity
    uh_e1 = np.einsum('ij,j->i',v_nodes[:,0,:],h_op) # normal horizontal out of plane velocity
    up_e1 = np.einsum('ij,j->i',v_nodes[:,0,:],p_e1) # normal in plane velocity

    sigma_ut_e1 = np.trapz(np.abs(ut_e1)**2*S_zeta,omegas)
    sigma_uh_e1 = np.trapz(np.abs(uh_e1)**2*S_zeta,omegas)
    sigma_up_e1 = np.trapz(np.abs(up_e1)**2*S_zeta,omegas)

    tt_e1 = np.outer(t_e1,t_e1) # local tangential to global components transformation matrix
    pp_e1 = np.outer(p_e1,p_e1) # local normal inplane to global components transformation matrix

    M[0:3,0:3] += mden*L_e1/2*np.eye(3) # element 1 mass contribution

    A_e1 = 1025*np.pi/4*deq**2*L_e1/2*(Can*(hh_op+pp_e1) + Cat*tt_e1) # element 1 added mass contribution

    B_e1 = 0.5*1025*deq*L_e1/2*np.sqrt(8/np.pi)*(Cdn*(sigma_uh_e1*hh_op + sigma_up_e1*pp_e1) +
                                                 Cdt*sigma_ut_e1*tt_e1) # element 1 damping contribution 

    K_e1 = EA/L_e1*tt_e1 + T_mean[0]/L_e1*(hh_op+pp_e1) # element 1 stiffness (axial + geometric)
    
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
        dr_bw = r_mean[n] - r_mean[n-1]
        L_bw = la.norm(dr_bw) # element 1 length
        t_bw = (dr_bw)/L_bw # tangential unit vector
        p_bw = np.cross(t_bw,h_op) # in plane normal unit vector

        ut_bw = np.einsum('ij,j->i',v_nodes[:,n,:],t_bw) # tangential velocity
        uh_bw = np.einsum('ij,j->i',v_nodes[:,n,:],h_op) # normal horizontal out of plane velocity
        up_bw = np.einsum('ij,j->i',v_nodes[:,n,:],p_bw) # normal in plane velocity

        sigma_ut_bw = np.trapz(np.abs(ut_bw)**2*S_zeta,omegas)
        sigma_uh_bw = np.trapz(np.abs(uh_bw)**2*S_zeta,omegas)
        sigma_up_bw = np.trapz(np.abs(up_bw)**2*S_zeta,omegas)

        tt_bw = np.outer(t_bw,t_bw) # local tangential to global components transformation matrix
        pp_bw = np.outer(p_bw,p_bw) # local normal inplane to global components transformation matrix

        M[3*n:3*n+3,3*n:3*n+3] += mden*L_bw/2*np.eye(3) # mass contribution from adjacent elements

        A_bw = 1025*np.pi/4*deq**2*L_bw/2*(Can*(hh_op+pp_bw) + Cat*tt_bw) # backward element added mass contribution

        B_bw = 0.5*1025*deq*L_bw/2*np.sqrt(8/np.pi)*(Cdn*(sigma_uh_bw*hh_op + sigma_up_bw*pp_bw) +
                                                     Cdt*sigma_ut_bw*tt_bw) # backward element damping contribution 

        K_bw = EA/L_bw*tt_bw + T_mean[0]/L_bw*(hh_op+pp_bw) # backward element stiffness (axial + geometric)

        ## forward element (n+1/2) contributions
        dr_fw = r_mean[n+1] - r_mean[n]
        L_fw = la.norm(dr_fw) # element 1 length
        t_fw = (dr_fw)/L_fw # tangential unit vector
        p_fw = np.cross(t_fw,h_op) # in plane normal unit vector

        ut_fw = np.einsum('ij,j->i',v_nodes[:,n,:],t_fw) # tangential velocity
        uh_fw = np.einsum('ij,j->i',v_nodes[:,n,:],h_op) # normal horizontal out of plane velocity
        up_fw = np.einsum('ij,j->i',v_nodes[:,n,:],p_fw) # normal in plane velocity

        sigma_ut_fw = np.trapz(np.abs(ut_fw)**2*S_zeta,omegas)
        sigma_uh_fw = np.trapz(np.abs(uh_fw)**2*S_zeta,omegas)
        sigma_up_fw = np.trapz(np.abs(up_fw)**2*S_zeta,omegas)

        tt_fw = np.outer(t_fw,t_fw) # local tangential to global components transformation matrix
        pp_fw = np.outer(p_fw,p_fw) # local normal inplane to global components transformation matrix
        
        M[3*n:3*n+3,3*n:3*n+3] += mden*L_fw/2*np.eye(3) # mass contribution from adjacent elements

        A_fw = 1025*np.pi/4*deq**2*L_fw/2*(Can*(hh_op+pp_fw) + Cat*tt_fw) # backward element added mass contribution

        B_fw = 0.5*1025*deq*L_fw/2*np.sqrt(8/np.pi)*(Cdn*(sigma_uh_fw*hh_op + sigma_up_fw*pp_fw) +
                                                     Cdt*sigma_ut_fw*tt_fw) # backward element damping contribution 

        K_fw = EA/L_fw*tt_fw + T_mean[0]/L_fw*(hh_op+pp_fw) # backward element stiffness (axial + geometric)

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
    dr_eN = r_mean[1] - r_mean[0]
    L_eN = la.norm(dr_eN) # element 1 length
    t_eN = (dr_eN)/L_eN # tangential unit vector
    p_eN = np.cross(t_eN,h_op) # in plane normal unit vector

    ut_eN = np.einsum('ij,j->i',v_nodes[:,0,:],t_eN) # tangential velocity
    uh_eN = np.einsum('ij,j->i',v_nodes[:,0,:],h_op) # normal horizontal out of plane velocity
    up_eN = np.einsum('ij,j->i',v_nodes[:,0,:],p_eN) # normal in plane velocity

    sigma_ut_eN = np.trapz(np.abs(ut_eN)**2*S_zeta,omegas)
    sigma_uh_eN = np.trapz(np.abs(uh_eN)**2*S_zeta,omegas)
    sigma_up_eN = np.trapz(np.abs(up_eN)**2*S_zeta,omegas)

    tt_eN = np.outer(t_eN,t_eN) # local tangential to global components transformation matrix
    pp_eN = np.outer(p_eN,p_eN) # local normal inplane to global components transformation matrix

    M[3*(N-1):3*(N-1)+3,3*(N-1):3*(N-1)+3] += mden*L_eN/2*np.eye(3) # element 1 mass contribution

    A_eN = 1025*np.pi/4*deq**2*L_eN/2*(Can*(hh_op+pp_eN) + Cat*tt_eN) # element 1 added mass contribution

    B_eN = 0.5*1025*deq*L_eN/2*np.sqrt(8/np.pi)*(Cdn*(sigma_uh_eN*hh_op + sigma_up_eN*pp_eN) +
                                                 Cdt*sigma_ut_eN*tt_eN) # element 1 damping contribution 

    K_eN = EA/L_eN*tt_eN + T_mean[0]/L_eN*(hh_op+pp_eN) # element 1 stiffness (axial + geometric)
    
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
        