import torch 
import triton
import triton.language as tl 

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
