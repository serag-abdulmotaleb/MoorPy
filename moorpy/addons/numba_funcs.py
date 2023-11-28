import numba
import numpy as np
from numpy.linalg import solve

@numba.njit(parallel=True, cache=True)
def _motion_parallel_solver(omegas, H_XF, F_hydro, F_aero, I,
                            S_hydro1, S_hydro2, S_aero,
                            S_XXwf, S_XXlf, H_FX):

    for n in numba.prange(omegas.shape[0]):
        H_FX_n = np.ascontiguousarray(solve(H_XF[n, :, :], I))
        H_FX[n, :, :] = H_FX_n

        F_hydro_n = np.ascontiguousarray(F_hydro[n, :])
        F_aero_n = np.ascontiguousarray(F_aero[n, :])
        S_hydro1_n = np.ascontiguousarray(np.outer(F_hydro_n, np.conj(F_hydro_n)))
        S_hydro2_n = np.ascontiguousarray(S_hydro2[n, :, :])
        S_aero_n = np.ascontiguousarray(np.outer(F_aero_n, np.conj(F_aero_n)))

        H_FX_n_conj = np.conj(H_FX_n.T)
        S_XXwf_n = H_FX_n @ S_hydro1_n @ H_FX_n_conj
        S_XXlf_n = (
            H_FX_n @ S_hydro2_n @ H_FX_n_conj +
            H_FX_n @ S_aero_n @ H_FX_n_conj
        )

        S_hydro1[n, :, :] = S_hydro1_n
        S_aero[n, :, :] = S_aero_n
        S_XXwf[n, :, :] = S_XXwf_n
        S_XXlf[n, :, :] = S_XXlf_n

    return H_FX, S_XXwf, S_XXlf

@numba.njit(parallel=True, cache=True)
def _tension_parallel_solver(omegas,H,F,X):
    H_UK = H[:,3:-3,3:-3]
    for n in numba.prange(omegas.shape[0]):
        # H = np.ascontiguousarray(-omegas[n]**2*(M+A) + 1j*omegas[n]*B + K)
        H_UK_n = np.ascontiguousarray(H_UK[n,:,:])
        # H_fl_n = np.ascontiguousarray(H[n,3:-3,:3])
        # RAO_fl_n = np.ascontiguousarray(RAO_fl[n,:])
        # F_n = np.ascontiguousarray(-H_fl_n @ RAO_fl_n)
        F_n = np.ascontiguousarray(F[n,:])
        X[n,3:-3] = solve(H_UK_n,F_n)
    return X

# def _parallel_linalg_solve(A,b,x):
#     for n in numba.prange(A.shape[0]):
#         A_n
#         x = solve()