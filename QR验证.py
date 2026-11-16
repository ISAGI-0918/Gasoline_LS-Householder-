import numpy as np

def householder_qr(A):
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for k in range(n):
        x = R[k:m, k]
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        sigma = np.linalg.norm(x)
        if sigma == 0:
            raise np.linalg.LinAlgError("Rank deficient matrix")
        sign = np.sign(x[0]) if x[0] != 0 else 1
        v = x + sign * sigma * e1
        v = v / np.linalg.norm(v)
        H = np.eye(m - k) - 2 * np.outer(v, v)
        R[k:m, k:n] = H @ R[k:m, k:n]
        Q[:, k:m] = Q[:, k:m] @ H
    Q_red = Q[:, :n]
    R_red = R[:n, :n]
   
    D = np.diag(np.sign(np.diag(R_red)))
    Q_red = Q_red @ D
    R_red = D @ R_red

    return Q_red, R_red

def back_substitution(R, b):
    n = R.shape[0]
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (b[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    return x

def ls_householder(A, b):
    Q, R = householder_qr(A)
    theta = back_substitution(R, Q.T @ b)
    return theta, Q, R

#测试
ones=np.array([1,1,1,1])
t=np.array([-1,0,1,2])
A = np.column_stack([ones,t])
b=np.array([1,1,2,2])

theta, Q, R = ls_householder(A,b)
print("theta（拟合参数）：", theta)
print("Q:\n", Q)
print("R:\n", R)
print("Q@R 复原A:\n", Q @ R)
theta_np = np.linalg.lstsq(A,b,rcond=None)[0]
print("参数 = ", theta_np)
