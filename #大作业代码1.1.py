import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== Householder QR (reduced QR) =====================
def householder_qr(A):
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for k in range(n):
        # 取当前子向量
        x = R[k:m, k]
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        sigma = np.linalg.norm(x)#用于构造 Householder 向量 v = x ± sigma·e1
        if sigma == 0:
            raise np.linalg.LinAlgError("Rank deficient matrix")
        # 符号选择，减少抵消：sign(x[0])
        sign = np.sign(x[0]) if x[0] != 0 else 1
        v = x + sign * sigma * e1
        v = v / np.linalg.norm(v)
        # Householder反射 H = I - 2vv^T
        H = np.eye(m - k) - 2 * np.outer(v, v)
        # 作用在R和Q上
        R[k:m, k:n] = H @ R[k:m, k:n]
        Q[:, k:m] = Q[:, k:m] @ H
    # reduced QR: Q(m,n), R(n,n)
    Q_red = Q[:, :n]
    R_red = R[:n, :n]
    D = np.diag(np.sign(np.diag(R_red)))
    Q_red = Q_red @ D
    R_red = D @ R_red
    return Q_red, R_red

# 回代求解上三角 R x = b
def back_substitution(R, b):
    n = R.shape[0]
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (b[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    return x

# Householder最小二乘求解
def ls_householder(A, b):
    Q, R = householder_qr(A)
    theta = back_substitution(R, Q.T @ b)
    return theta, Q, R

# ===================== 读取数据（csv和py同一文件夹，直接文件名） =====================
df = pd.read_csv("gasoline_december_1993_2022.csv")
t_all = df["year"].values
y_all = df["dec_price_usd_per_gallon"].values

# 划分训练/测试
train_mask = t_all <= 2016
test_mask  = t_all > 2016
t_train, y_train = t_all[train_mask], y_all[train_mask]
t_test,  y_test  = t_all[test_mask],  y_all[test_mask]

# 缩放变换 z = (t - 2004.5)/11.5
def scale_t(t):
    return (t - 2004.5) / 11.5
z_train = scale_t(t_train)
z_test  = scale_t(t_test)

# ===================== Part C：直线、二次拟合（缩放坐标） =====================
# 线性模型 y = θ0 + θ1 z
A_lin = np.column_stack([np.ones_like(z_train), z_train])
theta_lin, _, _ = ls_householder(A_lin, y_train)

# 二次模型 y = θ0 + θ1 z + θ2 z²
A_quad = np.column_stack([np.ones_like(z_train), z_train, z_train**2])
theta_quad, _, _ = ls_householder(A_quad, y_train)

def pred_lin(z, th):
    return th[0] + th[1] * z
def pred_quad(z, th):
    return th[0] + th[1] * z + th[2] * z**2

yhat_train_lin = pred_lin(z_train, theta_lin)
yhat_train_quad = pred_quad(z_train, theta_quad)
yhat_test_lin  = pred_lin(z_test, theta_lin)
yhat_test_quad = pred_quad(z_test, theta_quad)

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat)**2))
def rss(y, yhat):
    return np.sum((y - yhat)**2)

print("===== Householder QR | Part C 拟合结果（缩放z坐标） =====")
print(f"Linear θ: {theta_lin.round(4)}")
print(f"Linear Train RSS:{rss(y_train,yhat_train_lin):.4f}, Train RMSE:{rmse(y_train,yhat_train_lin):.4f}, Test RMSE:{rmse(y_test,yhat_test_lin):.4f}")
print(f"Quadratic θ: {theta_quad.round(4)}")
print(f"Quadratic Train RSS:{rss(y_train,yhat_train_quad):.4f}, Train RMSE:{rmse(y_train,yhat_train_quad):.4f}, Test RMSE:{rmse(y_test,yhat_test_quad):.4f}")

# ===================== Part D Compare algorithms and scaling =====================
print("===== Part D Compare algorithms and scaling =====")
print("Six‑run comparison, raw basis [1,t,t²] vs scaled basis [1,z,z²]")

A_orig = np.column_stack([np.ones_like(t_train), t_train, t_train**2])
A_scaled = A_quad

# -------------------------- 定义工具函数 --------------------------
def back_sub(R, b):
    n = R.shape[0]
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (b[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i,i]
    return x

def forward_sub(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i,i]
    return x

def rmse_calc(y, yhat):
    return np.sqrt(np.mean((y-yhat)**2))

# f_ref: reference fit vector: library QR on scaled‑A (project requirement!)
Q_lib_scaled, R_lib_scaled = np.linalg.qr(A_scaled, mode="reduced")
theta_ref_scaled = back_sub(R_lib_scaled, Q_lib_scaled.T @ y_train)
f_ref = A_scaled @ theta_ref_scaled   # project‑specified reference fitted vector

def one_experiment(A, y, solver_mode):
    """
    solver_mode:
        "hh_self"  : self‑written Householder QR
        "qr_lib"   : library reduced QR + manual back substitution
        "normal_chol": normal equation via cholesky
    returns dict: condA, condA_sq, eta_QR, omega_QR, rmse_train, delta_fit
    eta_QR, omega_QR = np.nan for normal equation
    """
    m, n = A.shape
    condA = np.linalg.cond(A, 2)
    condA_sq = condA ** 2
    eta_QR = np.nan
    omega_QR = np.nan
    theta_hat = None

    if solver_mode == "hh_self":
        theta_hat, Qh, Rh = ls_householder(A, y)
        eta_QR = np.linalg.norm(A - Qh @ Rh, ord="fro") / np.linalg.norm(A, ord="fro")
        omega_QR = np.linalg.norm(Qh.T @ Qh - np.eye(n), ord="fro")

    elif solver_mode == "qr_lib":
        Qlib, Rlib = np.linalg.qr(A, mode="reduced")
        theta_hat = back_sub(Rlib, Qlib.T @ y)
        eta_QR = np.linalg.norm(A - Qlib @ Rlib, ord="fro") / np.linalg.norm(A, ord="fro")
        omega_QR = np.linalg.norm(Qlib.T @ Qlib - np.eye(n), ord="fro")

    elif solver_mode == "normal_chol":
        Gram = A.T @ A
        try:
            L = np.linalg.cholesky(Gram)
            temp = forward_sub(L, A.T @ y)
            theta_hat = back_sub(L.T, temp)
        except np.linalg.LinAlgError:
            # Cholesky fail for ill‑conditioned matrix
            return {"condA":condA,"condA_sq":condA_sq,
                    "eta_QR":np.nan,"omega_QR":np.nan,
                    "rmse_train":np.nan,"delta_fit":np.nan,"cholesky_failed":True}

    y_fit = A @ theta_hat
    rmse_train = rmse_calc(y, y_fit)
    delta_fit = np.linalg.norm(y_fit - f_ref, 2) / np.linalg.norm(f_ref, 2)
    return {
        "condA": condA,
        "condA_sq": condA_sq,
        "eta_QR": eta_QR,
        "omega_QR": omega_QR,
        "rmse_train": rmse_train,
        "delta_fit": delta_fit,
        "cholesky_failed": False
    }

# -------------------------- 6‑run experiments --------------------------
cases = [
    ("Raw basis [1,t,t²]", A_orig),
    ("Scaled basis [1,z,z²]", A_scaled)
]
solvers = [
    ("Self‑Householder", "hh_self"),
    ("Library‑QR", "qr_lib"),
    ("Normal‑Cholesky", "normal_chol")
]

for case_name, Amat in cases:
    print(f"\n==== {case_name} ====")
    for sol_name, sol_key in solvers:
        res = one_experiment(Amat, y_train, sol_key)
        if res["cholesky_failed"]:
            print(f"  {sol_name:18s}: CHOLESKY FACTORIZATION FAILED")
            continue
        print(f"  {sol_name:18s}: κ2(A)={res['condA']:.2e}, κ²={res['condA_sq']:.2e},"
              f" η_QR={res['eta_QR']:.2e}, ω={res['omega_QR']:.2e},"
              f" RMSE_train={res['rmse_train']:.4f}, δ_fit={res['delta_fit']:.2e}")



# ===================== 绘图 =====================
plt.figure(figsize=(10,5))
plt.plot(t_train, y_train, "ko", label="Train data 1993–2016")
plt.plot(t_test, y_test, "rs", label="Test data 2017–2022")
t_grid = np.linspace(1993, 2022, 200)
z_grid = scale_t(t_grid)
plt.plot(t_grid, pred_lin(z_grid, theta_lin), "b-", lw=2, label="Linear fit")
plt.plot(t_grid, pred_quad(z_grid, theta_quad), "g-", lw=2, label="Quadratic fit")
plt.xlabel("Year")
plt.ylabel("Gasoline Price")
plt.title("Householder QR Least Squares Fit")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hh_fit.png", dpi=300)
plt.show()

plt.figure(figsize=(10,4))
res_lin  = y_train - yhat_train_lin
res_quad = y_train - yhat_train_quad
plt.plot(t_train, res_lin, "b-o", label="Linear residual")
plt.plot(t_train, res_quad, "g-s", label="Quadratic residual")
plt.axhline(0, c="k", ls="--")
plt.xlabel("Year")
plt.ylabel("Residual")
plt.title("Training Residuals (Householder QR)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hh_residual.png", dpi=300)
plt.show()
