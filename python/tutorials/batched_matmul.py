import torch

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    q_ptr,
    k_ptr,
    o_ptr,
    M,
    N,
    K,
    stride_qm,
    stride_qk,  #
    stride_kn,
    stride_kk,  #
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  #
    NUM_WARPS: tl.constexpr,
):
    # pid = tl.program_id(0)
    offs_b = tl.arange(0, NUM_WARPS)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N // NUM_WARPS)
    offs_k = tl.arange(0, BLOCK_K)
    stride_kb = BLOCK_N // NUM_WARPS
    q_ptrs1 = offs_b[:, None, None] * 0 + offs_m[None, :, None] * stride_qm + offs_k[None, None, :] * stride_qk
    q_ptrs = q_ptr + q_ptrs1
    q = tl.load(q_ptrs)
    k_ptrs = k_ptr + offs_b[:, None, None] * stride_kb + offs_n[None, None, :] * stride_kk + offs_k[None, :,
                                                                                                    None] * stride_kn

    qk = tl.zeros((NUM_WARPS, BLOCK_M, BLOCK_N // NUM_WARPS), dtype=tl.float32)
    k = tl.load(k_ptrs)
    qk = tl.dot(q, k)

    o_ptrs = o_ptr + offs_b[:, None, None] * (
        BLOCK_N // NUM_WARPS) + offs_m[None, :, None] * stride_om + offs_n[None, None, :] * stride_on
    tl.store(o_ptrs, qk)


def matmul(q, k):
    # Check constraints.
    assert q.shape[-1] == k.shape[0], "Incompatible dimensions"
    assert q.is_contiguous(), "Matrix A must be contiguous"
    assert k.is_contiguous(), "Matrix B must be contiguous"
    M, K = q.shape
    K, N = k.shape
    # Allocates output.
    o = torch.empty((M, N), device=q.device, dtype=q.dtype)

    BLOCK_M = 16
    BLOCK_N = 32
    BLOCK_K = 16
    num_warps = 2
    grid = (
        triton.cdiv(M, BLOCK_M),
        1,
    )
    matmul_kernel[grid](
        q,
        k,
        o,  #
        M,
        N,
        K,  #
        q.stride(0),
        q.stride(1),  #
        k.stride(0),
        k.stride(1),  #
        o.stride(0),
        o.stride(1),  #
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return o


#  q:  [16, 64], k: [4, 512, 64], v: [4, 512, 64]
# qk:  [4, 16, 512]
# qkv: [4, 16, 64]
M, N, K = 16, 32, 16
# torch.manual_seed(0)
# q = torch.eye(M, device='cuda', dtype=torch.float16)
# q_list = [[i for i in range(M)] for j in range(K)]
# q = torch.tensor(q_list, device='cuda', dtype=torch.float16)
q = torch.randn((16, 16), device='cuda', dtype=torch.float16)
# k_list = [[1 if (i % 16) == j else 0 for i in range(N)] for j in range(K)]
# k = torch.tensor(k_list, device='cuda', dtype=torch.float16)
k = torch.randn((16, 32), device='cuda', dtype=torch.float16)

triton_output = matmul(q, k)
torch_output = torch.matmul(q, k)
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
