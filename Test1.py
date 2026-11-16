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

# ===================== Part D：原始t基 vs 缩放z基 稳定性对比 =====================
print("\n===== Householder QR | Part D 条件数与稳定性 =====")
A_orig = np.column_stack([np.ones_like(t_train), t_train, t_train**2])
A_scaled = A_quad

def run_house_case(A, y):
    th_hh, Qh, Rh = ls_householder(A, y)
    # 参考解
    th_ref = np.linalg.lstsq(A, y, rcond=None)[0]
    delta = np.linalg.norm(th_hh - th_ref)
    omega = np.linalg.norm(Qh.T @ Qh - np.eye(Qh.shape[1]), 2)
    condA = np.linalg.cond(A, 2)
    return condA, omega, delta

cond_o, om_o, d_o = run_house_case(A_orig, y_train)
cond_s, om_s, d_s = run_house_case(A_scaled, y_train)

print(f"原始基 [1,t,t²]: κ2(A)={cond_o:.2e}, 正交损失ω={om_o:.2e}, 解误差δ={d_o:.2e}")
print(f"缩放基 [1,z,z²]: κ2(A)={cond_s:.2e}, 正交损失ω={om_s:.2e}, 解误差δ={d_s:.2e}")

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
