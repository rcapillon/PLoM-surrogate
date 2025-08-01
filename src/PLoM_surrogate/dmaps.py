import numpy as np


def gaussian_kernel(vec1, vec2, eps):
    k = (-1. / (4 * eps)) * np.linalg.norm(vec1 - vec2) ** 2

    return k


def build_mat_K(mat_eta, eps):
    N = mat_eta.shape[1]
    mat_K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            mat_K[i, j] = gaussian_kernel(mat_eta[:, i], mat_eta[:, j], eps)

    return mat_K


def build_mat_b(mat_K):
    N = mat_K.shape[0]
    mat_b = np.zeros((N, N))
    for i in range(N):
        mat_b[i, i] = np.sum(mat_K[i, :])

    return mat_b


def build_mat_Ps(mat_b, mat_K):
    inv_sqrt_b = np.linalg.inv(np.sqrt(mat_b))
    mat_Ps = np.dot(inv_sqrt_b, np.dot(mat_K, inv_sqrt_b))

    return mat_Ps


def construct_dmaps_basis(mat_eta, eps, m, kappa):
    mat_K = build_mat_K(mat_eta, eps)
    mat_b = build_mat_b(mat_K)
    mat_Ps = build_mat_Ps(mat_b, mat_K)

    eigvals, eigvects = np.linalg.eig(mat_Ps)
    inds_sort_eigvals = np.flip(np.argsort(eigvals))
    sorted_eigvals = S[inds_sort_eigvals]
    sorted_eigvects = eigvects[:, inds_sort_eigvals]

    vec_lambda = sorted_eigvals[:m]
    mat_phi = sorted_eigvects[:, :m]

    inv_sqrt_b = np.linalg.inv(np.sqrt(mat_b))
    mat_psi = np.dot(inv_sqrt_b, mat_phi)

    mat_g = np.zeros((mat_psi.shape[0], m))
    for i in range(m):
        mat_g[:, i] = (vec_lambda[i] ** kappa) * mat_psi[:, i]

    return mat_g


def build_mat_a(mat_g):
    inv_gTg = np.linalg.inv(np.dot(mat_g.T, mat_g))
    mat_a = np.dot(mat_g, inv_gTg)

    return mat_a


def project_on_dmaps(mat, mat_a):
    projected_mat = np.dot(mat, mat_a)

    return projected_mat


def reverse_project_from_dmaps(projected_mat, mat_g):
    mat = np.dot(projected_mat, mat_g.T)

    return mat