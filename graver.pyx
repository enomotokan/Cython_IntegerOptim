import numpy as np
cimport numpy as cnp
from time import time
from cython cimport boundscheck, wraparound
from cython.parallel cimport prange

cython: boundscheck=False
cython: wraparound=False

# 整基底を見つけるアルゴリズムの中で使うユークリッド互除法アルゴリズム

cdef int[:, ::1] euqlid(int[:, ::1] A, int i, int j, int k):
    cdef:
        int n = 0
        int m, coef
    if A[i, j] == 0:
        if A[i, k] != 0:
            for m in range(len(A)):
                A[m, j], A[m, k] = A[m, k], A[m, j]
    if A[i, k] == 0:
        pass
    elif A[i, j] >= A[i, k]:
        while True:
            if n == 0:
                coef = <int>(A[i, j] / A[i, k])
                for m in range(len(A)):
                    A[m, j] -= coef * A[m, k]
                if A[i, j] == 0:
                    for m in range(len(A)):
                        A[m, j], A[m, k] = A[m, k], A[m, j]
                    break
            else:
                coef = <int>(A[i, k] / A[i, j])
                for m in range(len(A)):
                    A[m, k] -= coef * A[m, j]
                if A[i, k] == 0:
                    break
            n += 1
            n %= 2
    else:
        while True:
            if n == 0:
                coef =<int>(A[i, k] / A[i, j])
                for m in range(len(A)):
                    A[m, k] -= coef * A[m, j]
                if A[i, k] == 0:
                    break
            else:
                coef = <int>(A[i, j] / A[i, k])
                for m in range(len(A)):
                    A[m, j] -= coef * A[m, k]
                if A[i, j] == 0:
                    for m in range(len(A)):
                        A[m, j], A[m, k] = A[m, k], A[m, j]
                    break
            n += 1
            n %= 2
    return A

# 入力を検証する
cdef bint verify_input(cnp.ndarray[cnp.npy_int32, ndim=2] A):
    if type(A) != np.ndarray:
        print("Error: Input is not Numpyarray")
        return False
    else:
        if A.dtype != int:
            print("Error: Dtype of input must be int")
            return False
        else:
            if A.ndim != 2:
                print("Error: Dimension of input matrix must be two")
                return False
            else:
                if A.shape[1] <= A.shape[0]:
                    print("Error: Width of input matrix must be larger than length")
                    return False
                else:
                    return True

# Aの核の格子点の整基底を計算する
def basis_intker(A):
    return c_basis_intker(A)

cdef cnp.ndarray[cnp.npy_int32, ndim=2] c_basis_intker(int[:, ::1] A):
    cdef:
        int N = len(A)
        int n_deleted = 0
        int i, j, k
        bint equal_0
        int equal_1
    
    if verify_input(np.asarray(A)):
        A = np.append(A, np.eye(A.shape[1]).astype(np.int32), axis=0)
        for i in range(N):
            equal_0 = True
            for j in range(len(A[i - n_deleted][i - n_deleted:])):
                if A[i - n_deleted][i - n_deleted + j] != 0:
                    equal_0 = False
                    break
            if equal_0:
                A = np.delete(A , i - n_deleted)
                n_deleted += 1
                continue
            equal_1 = -1
            for j in range(len(A[i - n_deleted, i - n_deleted:])):
                if A[i - n_deleted, i - n_deleted + j] == 1:
                    equal_1 = j
                    break
            if equal_1 == -1:
                for j in range(i - n_deleted + 1, A.shape[1]):
                    if A[i, j] < -1:
                        for k in range(A.shape[0]):
                            A[k, j] *= -1
                    A = np.asarray(euqlid(A, i - n_deleted, i - n_deleted, j))
            else:
                A[:, i - n_deleted], A[:, i - n_deleted + equal_1] = A[:, i - n_deleted + equal_1].copy(), A[:, i - n_deleted].copy()
                for j in range(i - n_deleted + 1, A.shape[1]):
                    for k in range(A.shape[0]):
                        A[k, j] -= A[i, j] * A[k, i - n_deleted]
        
        return np.asarray(A[i - n_deleted + 1:, i - n_deleted + 1:]).T

# 二分探索    
cdef int binary_search(cnp.ndarray[cnp.npy_int32, ndim=1] norms, cnp.ndarray[cnp.npy_int32, ndim=1] range, int norm):
    cdef int central
    if range[1] - range[0] == 0:
        return range[0]
    elif range[1] - range[0] == 1:
        if norms[range[1]] == norm:
            return range[1]
        else:
            return range[0]
    else:
        central = int((range[1] + range[0]) / 2)
        if norms[central] < norm:
            return binary_search(norms, np.array([central + 1, range[1]]), norm)
        else:
            return binary_search(norms, np.array([range[0], central]), norm)

# 標準形アルゴリズム
cdef int[:] NFA(int[:] s, int[:, ::1] G):
    cdef:
        bint equal_0 = True, notlarger
        int n, coef
    for g in G:
        for n in range(len(s)):
            if s[n] != 0:
                equal_0 = False
        if equal_0:
            break
        notlarger = True
        for n in range(len(s)):
            if np.sign(g[n]) * np.sign(s[n]) < 0 or np.abs(g[n]) > np.abs(s[n]):
                notlarger = False
                break
        # if np.all(np.sign(g) * np.sign(s) >= 0):
        #     if np.all(np.abs(g) <= np.abs(s)):
        if notlarger:
            coef = 2**20
            for n in range(len(s)):
                if g[n] != 0:
                    coef = np.min([coef, <int>(s[n] / g[n])])
            for n in range(len(s)):
                s[n] -= coef * g[n]
    return s

# 与えられた整基底の生成する加群のグレーバー基底を計算する

def graver(basis):
    if verify_input(basis):
        return c_graver(basis)

cdef cnp.ndarray[cnp.npy_int32, ndim=2] c_graver(cnp.ndarray[cnp.npy_int32, ndim=2] basis):

    cdef:
        float[:, ::1] M = basis.copy().astype(np.float32)
        int[:] linear_independent = np.array([], dtype=np.int32)
        int i, j, k, l

    for i in range(M.shape[0]):
        equal_0 = np.abs(M[i]) <= 2**-15
        if np.all(equal_0):
            M[i] = np.zeros(M.shape[1])
            continue
        else:
            k = np.arange(M.shape[1])[np.logical_not(equal_0)][0]
            for j in np.delete(np.arange(M.shape[1]), k):
                M[i, j] /= M[i, k]
            M[i, k] = 1
            for j in np.delete(np.arange(M.shape[0]), i):
                for l in np.delete(np.arange(M.shape[1]), k):
                    M[j, l] -= M[i, l] * M[j, k]
                M[j, k] = 0
            linear_independent = np.append(linear_independent, k)
            if len(linear_independent) == M.shape[0]:
                break

    cdef cnp.ndarray[cnp.npy_int32, ndim=2] projected_basis = basis[:, [i in linear_independent for i in range(basis.shape[1])]].copy()

    # 射影された基底に対してPottierアルゴリズムを適用する
    cdef:
        cnp.ndarray[cnp.npy_int32, ndim=2] G = np.array([projected_basis[np.int32(n / 2)] if n % 2 == 0 else - projected_basis[np.int32((n - 1) / 2)] for n in range(2 * projected_basis.shape[0])])
        cnp.ndarray[cnp.npy_int32, ndim=2] C = np.empty((0, projected_basis.shape[1]), dtype=np.int32)
        cnp.ndarray[cnp.npy_int32, ndim=1] s
        cnp.ndarray[cnp.npy_int32, ndim=1] g

    for i in range(projected_basis.shape[0]):
        for j in range(i + 1, projected_basis.shape[0]):
            C = np.append(C, [projected_basis[i] + projected_basis[j]], axis=0)
            C = np.append(C, [- projected_basis[i] - projected_basis[j]], axis=0)
            C = np.append(C, [projected_basis[i] - projected_basis[j]], axis=0)
            C = np.append(C, [- projected_basis[i] + projected_basis[j]], axis=0)

    while len(C) != 0:
        s = np.asarray(NFA(C[0], G))
        C = np.delete(C, 0, axis=0)
        if not np.all(s == 0):
            for j in range(len(G)):
                C = np.append(C, [np.add(G[j], s)], axis=0)
            G = np.append(G, [s], axis=0)
    
    # 極小元でない元を除く
    cdef:
        cnp.ndarray[cnp.npy_int32, ndim=2] G_ = np.empty((0, G.shape[1]), dtype=np.int32)
        bint minimum = True
        cnp.ndarray[cnp.npy_int32, ndim=1] not_minimum_list = np.array([], dtype=np.int32)
    while len(G) != 0:
        minimum = True
        for j in range(1, len(G)):
            if np.all(np.sign(G[0]) * np.sign(G[j]) >= 0):
                if np.all(np.abs(G[0]) >= np.abs(G[j])) and not np.all(np.abs(G[0]) == np.abs(G[j])):
                    minimum = False
                elif np.all(np.abs(G[0]) <= np.abs(G[j])):
                    not_minimum_list = np.append(not_minimum_list, j)
        if minimum:
            G_ = np.append(G_, [G[0]], axis=0)
        G = np.delete(G, not_minimum_list, axis=0)
        not_minimum_list = np.array([], dtype=np.int32)
        G = np.delete(G, 0, axis=0)
    
    # 射影・持ち上げアルゴリズム
    cdef:
        cnp.ndarray[cnp.npy_float32, ndim=2] projected_basis_inv = np.linalg.inv(projected_basis.astype(np.float32))
        cnp.ndarray[cnp.npy_bool, cast=True, ndim=1] notin_linearindependent = np.array([i not in linear_independent for i in range(basis.shape[1])], dtype=bool)
        cnp.ndarray[cnp.npy_int32, ndim=2] F_notin_linearindependent = np.round(np.dot(np.dot(G_, projected_basis_inv), basis[:, notin_linearindependent])).astype(np.int32)
        cnp.ndarray[cnp.npy_int32, ndim=2] F = np.empty((G_.shape[0], basis.shape[1]), dtype=np.int32)
        int n_added_in_linearindependent = 0
        int n_added_notin_linearindependent = 0

    for i in range(basis.shape[1]):
        if i in linear_independent:
            F[:, i] = G_[:, n_added_in_linearindependent]
            n_added_in_linearindependent += 1
        else:
            F[:, i] = F_notin_linearindependent[:, n_added_notin_linearindependent]
            n_added_notin_linearindependent += 1

    G = F.copy()
    C = np.empty((0, G.shape[1]), dtype=np.int32)

    for i in range(G.shape[0]):
        for j in range(i + 1, G.shape[0]):
            if not np.all(G[i] == -G[j]):
                C = np.append(C, [G[i] + G[j]], axis=0)
    
    cdef:
        cnp.ndarray[cnp.npy_bool, cast=True, ndim=1] project = np.array([i in linear_independent for i in range(basis.shape[1])], dtype=np.bool_)
        cnp.ndarray[cnp.npy_int32, ndim=2] norm_index = np.empty((0, 2), dtype=np.int32)
        cnp.ndarray[cnp.npy_int32, ndim=1] norms = np.array([], dtype=np.int32)
        cnp.ndarray[cnp.npy_int32, ndim=1] sorted_norm_index
    for c in C:
            norms = np.append(norms, np.sum(np.abs(c)).astype(np.int32))
    sorted_norm_index =np.argsort(norms).astype(np.int32)
    C = C[sorted_norm_index]
    cdef:
        int n = 0
        int norm_ = 0
    for norm in norms[sorted_norm_index]:
        if norm_ != norm:
            norm_index = np.append(norm_index, [[norm, n]], axis=0)
        norm_ = norm
        n += 1

    cdef:
        cnp.ndarray[cnp.npy_int32, ndim=1] sign_s
    
    while True:
        if len(C) == 0:
            break
        s = C[0]
        C = np.delete(C, 0, axis=0)
        if len(norm_index) >= 2:
            norm_index[1:, 1] -= 1
            if norm_index[1, 1] == 0:
                norm_index = np.delete(norm_index, 0, axis=0)

        sign_s = np.sign(s)
        # if not np.any([np.all(np.sign(v) * sign_s >= 0) and np.all(v <= s) for v in G]): 
        if np.all([np.any(np.sign(v) * sign_s < 0) or np.any(np.abs(v) > np.abs(s)) for v in G]):
            for g in G:
                if np.all(np.sign(s[project]) * np.sign(g[project]) >= 0):
                    norm = np.sum(np.abs(s + g))
                    if norm_index[-1][0] < norm:
                        norm_index = np.append(norm_index, np.array([[norm, len(C)]]), axis=0)
                        C = np.append(C, [np.add(s, g)], axis=0)
                    else:
                        i = binary_search(norm_index[:, 0], np.array([0, len(norm_index) - 1]), norm)
                        C = np.insert(C, norm_index[i][1], [np.add(s, g)], axis=0)
                        if norm < norm_index[i][0]:
                            norm_index = np.insert(norm_index, i, [[norm, norm_index[i][1]]], axis=0)
                        if i != len(norm_index) - 1:
                            norm_index[i + 1:][:, 1] += 1
            G = np.append(G, [s], axis=0)
    return G

# if __name__ == "__main__":
A = np.array([[2, -1, 0, -3, 2, -2], [1, 5, -4, 0, 0, 0]], dtype=np.int32)
t = time()
basis = basis_intker(A)
print(basis)
print(graver(basis))
# print(len(proj_lift))
t1 = time()
print(t1 - t)