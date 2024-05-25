import torch
# 定义共轭梯度法来解决 Ax = b
def conjugate_gradient(Hv_func, b, nsteps, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()  # r = b - A*x (x 初始化为 0, A*x = 0)
    p = r.clone()  # 初始搜索方向和残差相同
    rr = torch.dot(r, r)  # r 的二范数的平方
    for i in range(nsteps):
        Ap = Hv_func(p)  # A*p
        pAp = torch.dot(p, Ap)  # p^T * A * p
        alpha = rr / pAp  # 步长
        x += alpha * p  # 更新解
        r -= alpha * Ap  # 更新残差
        new_rr = torch.dot(r, r)  # 计算新的残差的二范数平方
        if new_rr < tol:  # 检查收敛
            break
        beta = new_rr / rr  # 更新方向的系数
        p = r + beta * p  # 更新搜索方向
        rr = new_rr  # 更新残差的范数
    return x

# 假设 theta 和 alpha 是模型参数和超参数
theta = torch.randn(10, requires_grad=True)
alpha = torch.randn(5, requires_grad=True)

# 定义损失函数
def total_loss(theta, alpha):
    return torch.sum(theta**2 + alpha**2)  # 示例总损失函数

def cls_loss(theta):
    return torch.sum(theta**2)  # 示例分类损失函数

# 计算 L_cls 对 theta 的梯度
grad_cls_theta = torch.autograd.grad(cls_loss(theta), theta, create_graph=True)[0]

# 使用 hessian_vector_product 函数来计算 Hessian 向量积
def hessian_vector_product(loss, params, vector):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_vector_product = torch.sum(torch.stack([torch.sum(g * v) for g, v in zip(grads, vector)]))
    hvp = torch.autograd.grad(grad_vector_product, params, retain_graph=True)
    return hvp

# 使用 conjugate_gradient 函数计算逆 Hessian 乘以 grad_cls_theta
inv_Hv = conjugate_gradient(lambda v: hessian_vector_product(total_loss(theta, alpha), [theta], [v])[0], grad_cls_theta, 10)

# 计算混合二阶导数 nabla^2_(alpha,theta) L_Total
grad_total_alpha_theta = torch.autograd.grad(total_loss(theta, alpha), (alpha, theta), create_graph=True)

# 计算最终梯度 -nabla^2_(alpha,theta) L_Total * inv_Hv
final_grad = -torch.matmul(grad_total_alpha_theta[1], inv_Hv)

print(final_grad)
