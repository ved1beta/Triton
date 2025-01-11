import torch 
import triton
import triton.language as tl 

@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    
class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, casual , softmax_scale ):
        HEAD_DIM_Q , HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V =  V.shape[-1]
        BATCH_SIZE , NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)

        stage = 3 if casual else 1 
        grid = lambda agrs:(

        torch.cdiv(SEQ_LEN, agrs["BLOCK_SIZE_Q"]), 
        BATCH_SIZE*NUM_HEADS,
        1
        )
        M = torch.empty(
        (BATCH_SIZE, NUM_HEADS ,SEQ_LEN), device= Q.device, dtype=torch.float32
        )

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = casual
        return O



def tes_op(BATCH_SIZE , NUM_HEADS ,  SEQ_LEN, HEAD_DIM, casual , dtype = torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std= 0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std= 0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std= 0.5)
        .requires_grad_()
    )
    
    softmax_scale = 1 /(HEAD_DIM**0.5) 
    do = torch.rand_like(Q)
# MASK
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device = "cuda"))
    P = torch.matmul(Q, K.transpose(2,3))*softmax_scale
    if casual:
        P[:, :, MASK ==0 ] = float("-inf")
        P= torch.softmax(P.float(),dim = -1 ).half()
        ref_0= torch.matmul(P,V)
        ref_0.backward(do)

        ref_dV , V.grad = V.grad.clone(), None
        ref_dK , K.grad = K.grad.clone(), None
        ref_dQ , Q.grad = Q.grad.clone(), None

        tri_out = TritonAttention.apply(Q, K, V, casual, softmax_scale).half()
        tri_out.backward(do)
        ref_dV , V.grad = V.grad.clone(), None
        ref_dK , K.grad = K.grad.clone(), None
        ref_dQ , Q.grad = Q.grad.clone(), None

