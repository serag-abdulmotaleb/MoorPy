# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:42:41 2023

@author: seragela
"""

import numpy as np
import moorpy as mp

def spread_mooring(moor_dict, tol = 0.01, maxIter = 500, no_fail = True, finite_difference = False):
    """
    Creates a moorpy mooring system object with axis-symmetric, multi-segmented line configuration for multi-column platform.
    (N.B. The platform body type in this system is set to fixed in position so that no information about platform's mass or 
     buoyancy is required. The body type can be changed in subsequent analysis once the mass and buoyancy information are added.)

    Parameters
    ----------
    moor_dict : dictionary
        A dictionary with the following keys:
            nCols: int
                number of outer columns of the platform.
            legsPerCol: int
                number of mooring legs attached to a single column
            colAngles: list of floats
                oreintation of outer columns in degrees
            spreadAngle: float
                angle on which the legs in a single column are spread in degrees (0 if 1 leg per column)
            depth: float
                seabed depth
            rCol: float
                distance between an outer column center and platfomr center
            Dcol: float
                column diameter
            lF2A: float
                fairlead to anchor distance
            zFair: float
                fairlead vertical position w.r.t SWL
            nLines: int
                number of lines per mooring leg
            lineLengths: list of floats
                line lengths in meters
            segLengths: list of floats
                segment lengths in meters
            lineDiameters: list of floats
                line nominal diameters in meters
            lineTypes: list of strings
                line material type (must match the name key of one of the types in the lineTypes input)
            materialDicts: list of dictionaries
                line type material dictionaries each with the following keys:
                    name: (str) type name. e.g. chain, wire etc.
                    massden: (float) linear density in kg/m
                    EA: (float) extensional stiffness in N/m
                    d: (float) volume equivalent diameter in m
                    MBL: (float) minimum breaking load in N

    Returns
    -------
    conv : bool
        moorpy convergence flag.
    ms : moorpy.System
        A moorpy mooring System object.
    """
    # TODO: make line discretization customizable
    # Read parameters from input dictionary
    n_cols = moor_dict['nCols']
    n_legs = moor_dict['legsPerCol']
    col_angles = np.radians(moor_dict['colAngles'])
    if n_legs > 1:
        spread_angle =  np.radians(moor_dict['spreadAngle'])
    else:
        spread_angle = 0
    depth = moor_dict['depth']
    r_col = moor_dict['rCol']
    D_col = moor_dict['Dcol']
    l_f2a = moor_dict['lF2A']
    z_fair = moor_dict['zFair']
    n_lines = moor_dict['nLines']
    line_lengths = moor_dict['lineLengths'] 
    seg_lengths = moor_dict['segLengths']
    line_diameters = moor_dict['lineDiameters'] 
    line_types = moor_dict['lineTypes']
    material_dicts = moor_dict['materialDicts'] 
    L_tot = np.sum(line_lengths)

    # TODO: ADD ERROR HANDLING
    
    # Create mooring system
    ms = mp.System(depth = depth)
    
    # Add line types
    for lt in material_dicts:
        ms.lineTypes[lt['name']] = mp.LineType(name = lt['name'], massden = lt['massDensity'], EA = lt['EA'], d = lt['deq'], MBL = lt['MBL'],
                                               Can = lt['Can'], Cat = lt['Cat'], Cdn = lt['Cdn'], Cdt = lt['Cdt'])
    # Add a fixed body (to be changed to floating in subsequent analyses)
    ms.addBody(1, r6 = [0]*6)
    
    # Add connection points and lines
    for col, col_angle in enumerate(col_angles): # loop over platform's columns
        for n in range(n_legs): # loop over lines per column
            ## set line angle for evaluation of fairlead position    
            if n_legs > 1:
                line_angle = col_angle + spread_angle/2 - n*spread_angle/(n_legs-1) 
            else:
                line_angle = col_angle
            
            r_fairlead = np.array([r_col*np.cos(col_angle) + D_col/2*np.cos(line_angle), # fairlead x-coord
                                   r_col*np.sin(col_angle) + D_col/2*np.sin(line_angle), # fairlead y-coord
                                   z_fair])            # fairlead z-coord
            
            r_anchor = np.array([r_col*np.cos(col_angle) + (D_col/2+l_f2a)*np.cos(line_angle), # anchor x-coord
                                 r_col*np.sin(col_angle) + (D_col/2+l_f2a)*np.sin(line_angle), # anchor y-coord
                                 -depth]) # anchor z-coord
            
            ## Add fairlead point
            ms.addPoint(1, r_fairlead)                                   
            ms.bodyList[0].attachPoint(pointID = len(ms.pointList), rAttach = ms.pointList[len(ms.pointList)-1].r)

            ## Add intermediate connection points (if any)
            if n_lines>1: # loop over line segements
                r_0 = r_fairlead
                point_masses = moor_dict['clumpMasses'] # a list of intermediate clump masses (must be 1 less than the number of lines)
                point_volumes = moor_dict['buoyVolumes'] # a list of intermediate buoy volumes (must be 1 less than the number of lines)
                for line_len,lt,seg_len,m,v in zip(line_lengths[:-1],line_types[:-1], seg_lengths, point_masses, point_volumes):
                    r_0 = r_0 + line_len/L_tot*(r_anchor-r_fairlead)
                    n_segs = int(line_len/seg_len)
                    ms.addPoint(0, r_0, m = m, v = v)
                    ms.addLine(line_len, lt, nSegs=n_segs, pointA = len(ms.pointList)-1, pointB = len(ms.pointList))

            ## Add anchor point
            ms.addPoint(1, r_anchor)
            n_segs = int(line_lengths[-1]/seg_lengths[-1])
            ms.addLine(line_lengths[-1], line_types[-1],nSegs = n_segs, pointA = len(ms.pointList)-1, pointB = len(ms.pointList))
    
    ms.initialize()
    conv = ms.solveEquilibrium(tol = tol, no_fail = no_fail, maxIter = maxIter, finite_difference = finite_difference)
    

    return conv,ms