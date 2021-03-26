import numpy as np

def get_neighbors(size):
    # neighbors = np.zeros((size, size, 4, 2))
    neighbors = np.zeros((size*size, 4))
    site_col = np.tile(np.arange(size), size)
    site_row = np.repeat(np.arange(size), size)
    # left neighbor:
    neighbors[:, 0] = site_row*size + np.mod(site_col-1, size)
    # right neighbor:
    neighbors[:, 1] = site_row*size + np.mod(site_col+1, size)
    # top neighbor:
    neighbors[:, 2] = np.mod(site_row-1, size)*size + site_col
    # bot neighbor:
    neighbors[:, 3] = np.mod(site_row+1, size)*size + site_col
    return neighbors.astype(int)


def bi_pref_shift(state, sigma_A, sigma_B, shift_draw, a=1, mu=0.01, c=0.1,
                  q=0.5, s=0.5, rate=1, **kwargs):
    new_state = state
    proba_0_to_2 = rate * c * (1-mu) * (1-s) * (
        sigma_B + (1-q) * (1-sigma_A-sigma_B))**a
    if state == 0 and shift_draw < proba_0_to_2:
        new_state = 2

    elif state == 1 and (shift_draw < rate * c * (1-mu) * s * 
                                      (sigma_A + q * (1-sigma_A-sigma_B))**a):
        new_state = 2

    elif state == 2:
        if shift_draw < rate * mu * s * (sigma_A + q * (1-sigma_A-sigma_B))**a:
            new_state = 0
        elif shift_draw < (rate * mu * (1-s) *
                           (sigma_B + (1-q) * (1-sigma_A-sigma_B))**a):
            new_state = 1

    return new_state
