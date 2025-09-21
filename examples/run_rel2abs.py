#!/usr/bin/env python3
"""
这是相对位姿转换为绝对位姿的验证脚本
使用方法:
    python examples/run_rel2abs.py
"""

import torch

# ========== 工具函数 ==========
def quat_normalize(q):
    return q / torch.norm(q, dim=-1, keepdim=True)

def quat_mul(q, r):
    """四元数乘法 (w,x,y,z)"""
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def quat_inv(q):
    """单位四元数的逆 (w,-x,-y,-z)"""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def quat_to_rotmat(q):
    """四元数转旋转矩阵 (w,x,y,z)，支持批量"""
    w, x, y, z = q.unbind(-1)
    B = q.shape[0]

    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)
    ], dim=-1).reshape(B, 3, 3)
    return R

# ========== 转换函数 ==========
def absolute_to_relative_torch(pA, qA, pB, qB):
    """
    输入: pA,pB (N,3), qA,qB (N,4) (wxyz)
    输出: p_rel (N,3), q_rel (N,4)
    """
    RA = quat_to_rotmat(qA)
    RB = quat_to_rotmat(qB)

    p_rel = torch.bmm(RB.transpose(1,2), (pA - pB).unsqueeze(-1)).squeeze(-1)
    q_rel = quat_mul(quat_inv(qB), qA)
    return p_rel, quat_normalize(q_rel)

def relative_to_absolute_torch(p_rel, q_rel, pB, qB):
    """
    输入: p_rel (N,3), q_rel (N,4), pB (N,3), qB (N,4) (wxyz)
    输出: pA (N,3), qA (N,4)
    """
    RB = quat_to_rotmat(qB)
    pA = torch.bmm(RB, p_rel.unsqueeze(-1)).squeeze(-1) + pB
    qA = quat_mul(qB, q_rel)
    return pA, quat_normalize(qA)

# ========== 验证函数 ==========
def verify_round_trip_torch(batch_size=5, tol=1e-6, device="cpu"):
    pA = torch.randn(batch_size, 3, device=device)
    pB = torch.randn(batch_size, 3, device=device)

    qA = quat_normalize(torch.randn(batch_size, 4, device=device))
    qB = quat_normalize(torch.randn(batch_size, 4, device=device))

    # 绝对 -> 相对
    p_rel, q_rel = absolute_to_relative_torch(pA, qA, pB, qB)

    # 相对 -> 绝对
    pA_rec, qA_rec = relative_to_absolute_torch(p_rel, q_rel, pB, qB)

    # 误差
    pos_err = torch.norm(pA - pA_rec, dim=-1)
    quat_err = torch.min(
        torch.norm(qA - qA_rec, dim=-1),
        torch.norm(qA + qA_rec, dim=-1)  # 四元数符号不唯一
    )

    print("位置误差:", pos_err)
    print("四元数误差:", quat_err)

    assert torch.all(pos_err < tol), "位置还原失败！"
    assert torch.all(quat_err < tol), "姿态还原失败！"
    print("所有 PyTorch 测试通过 ✅")

# ==== 运行验证 ====
# print("开始验证...")
verify_round_trip_torch()
