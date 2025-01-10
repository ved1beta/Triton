import torch 
import triton
import triton.language as tl 

class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, casual , softmax_scale ):
        HEAD_DIM_Q , HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V =  V.shape[-1]
        BATCH_SIZE , NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)

        stage = 3 if casual else 1 
        grind = lambda agrs:(

        torch.cdiv(SEQ_LEN, agrs["BLOCK_SIZE_Q"]), 
        BATCH_SIZE*NUM_HEADS,
        1

        )



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

