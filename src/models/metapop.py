'''
Implements the computation of the time derivatives and associated Jacobian
corresponding to the approximated equations in a metapopulation. Added kwargs in
every function so that we may reuse the parameter dictionary used in the models,
even if some of the parameters it contains are not used in these functions.
'''
import numpy as np


def bi_model_system(N_L, N, nu, nu_T_N, a=1, s=0.5, rate=1, **kwargs):
    '''
    Computes the values of the time derivatives in every cell for the two
    monolingual kinds, for Castello's model.
    '''
    N_A = N_L[:N.shape[0]]
    N_B = N_L[N.shape[0]:]
    # Every element of the line i of nu must be divided by the same value
    # sigma[i], hence this trick with the two transpose.
    nu_T_N_A = np.dot(nu.T, N_A)
    nu_T_N_B = np.dot(nu.T, N_B)
    N_A_eq = rate * (
        s * (N - N_A - N_B) * np.dot(nu, (1 - nu_T_N_B / nu_T_N)**a)
        - (1-s) * N_A * np.dot(nu, (nu_T_N_B / nu_T_N)**a))
    N_B_eq = rate * (
        (1-s) * (N - N_A - N_B) * np.dot(nu, (1 - nu_T_N_A / nu_T_N)**a)
        - s * N_B * np.dot(nu, (nu_T_N_A / nu_T_N)**a))
    return np.concatenate((N_A_eq, N_B_eq))


def bi_pref_system(N_L, N, nu, nu_T_N, mu=0.02, c=0.1, s=0.5, q=0.5, rate=1,
                   **kwargs):
    '''
    Computes the values of the time derivatives in every cell for the two
    monolingual kinds, for our model.
    '''
    N_A = N_L[:N.shape[0]]
    N_B = N_L[N.shape[0]:]
    # Every element of the line i of nu must be divided by the same value
    # sigma[i], hence this trick with the two transpose.
    nu_T_N_A = np.dot(nu.T, N_A)
    nu_T_N_B = np.dot(nu.T, N_B)
    sum_nu_rows = np.sum(nu, axis=1)
    nu_nu_T_N_L_term = np.dot(nu, ((1-q)*nu_T_N_A - q*nu_T_N_B) / nu_T_N)
    N_A_eq = rate * (
        mu*s * (N - N_A - N_B) * (q*sum_nu_rows + nu_nu_T_N_L_term)
        - c*(1-mu)*(1-s) * N_A * ((1-q)*sum_nu_rows - nu_nu_T_N_L_term))
    N_B_eq = rate * (
        mu*(1-s) * (N - N_A - N_B) * ((1-q)*sum_nu_rows - nu_nu_T_N_L_term)
        - c*(1-mu)*s * N_B * (q*sum_nu_rows + nu_nu_T_N_L_term))
    return np.concatenate((N_A_eq, N_B_eq))


def bi_pref_jacobian(N_L, N, nu, nu_T_N, mu=0.02, c=0.1, s=0.5, q=0.5,
                     **kwargs):
    '''
    Computes the Jacobian of the system at a given point for our model.
    '''
    n_cells = N.shape[0]
    N_A = N_L[:n_cells]
    N_B = N_L[n_cells:]
    nu_T_N_A = np.dot(nu.T, N_A)
    nu_T_N_B = np.dot(nu.T, N_B)
    nu_cols_prod = np.dot(nu / nu_T_N, nu.T)
    nu_T_N_L_term = ((1-q)*nu_T_N_A - q*nu_T_N_B) / nu_T_N
    sum_nu_rows = np.sum(nu, axis=1)
    AA_block = ((mu*s*(1-q)*(N-N_A-N_B) + c*(1-mu)*(1-s)*(1-q)*N_A)
                * nu_cols_prod.T).T
    AA_block += np.eye(n_cells) * (
        (-mu*s*q - c*(1-mu)*(1-s)*(1-q)) * sum_nu_rows
        + np.dot(
            nu,
            (c*(1-mu)*(1-s) - mu*s) * nu_T_N_L_term))

    AB_block = ((-mu*s*q*(N-N_A-N_B) - c*(1-mu)*(1-s)*q*N_A)
                * nu_cols_prod.T).T
    AB_block += np.eye(n_cells) * (
        -mu*s*q * sum_nu_rows
        + np.dot(
            nu,
            -mu*s * nu_T_N_L_term))

    BA_block = (-(mu*(1-s)*(1-q)*(N-N_A-N_B) - c*(1-mu)*s*(1-q)*N_B)
                * nu_cols_prod.T).T
    BA_block += np.eye(n_cells) * (
        -mu*(1-s)*(1-q) * sum_nu_rows
        + np.dot(
            nu,
            mu*(1-s) * nu_T_N_L_term))

    BB_block = ((mu*(1-s)*q*(N-N_A-N_B) + c*(1-mu)*s*q*N_B)
                * nu_cols_prod.T).T
    BB_block += np.eye(n_cells) * (
        (-mu*(1-s)*(1-q) - c*(1-mu)*s*q) * sum_nu_rows
        + np.dot(
            nu,
            (-c*(1-mu)*s + mu*(1-s)) * nu_T_N_L_term))

    jacobian = np.block([[AA_block, AB_block],
                         [BA_block, BB_block]])
    return jacobian
