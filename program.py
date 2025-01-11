import torch 
import triton
import triton.language as tl 
@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        # From 0 to the left of the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # Only used for non-causal attention
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, 0, lo)
    V_block_ptr = tl.advance(V_block_ptr, lo, 0)

    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv,BLOCK_SIZE_KV)
        K_Block = tl.load(K_block_ptr)
        QK_Block = tl.dot(Q_block, K_Block)

    if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
    else:
            # Compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
    P_block = tl.math.exp(QK_block)
    # Compute the sum by rows of the attention scores
    l_ij = tl.sum(P_block, 1)
    # This is the correction factor for the previous l_i
    alpha = tl.math.exp(m_i - m_ij)
    # Apply the correction factor to the previous l_i and add the new l_ij
    l_i = l_i * alpha + l_ij
    V_block = tl.load(V_block_ptr)
    P_block = P_block.to(tl.float16)
    # This computes the following: O_new = P x V + O_old * alpha
    O_block = O_block * alpha[:, None]
    O_block = tl.dot(P_block, V_block, O_block)
    m_i = m_ij
    # Move to the next block of K and V
    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    return O_block, l_i, m_i

        


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
    block_index_q = tl.program_id()
    index_bacth_head = tl.program_id(1)

    index_bacth = index_bacth_head//NUM_HEADS

    index_head = index_bacth_head % NUM_HEADS

    qvk_offest = {
        index_bacth.to(tl.int64)*stride_Q_batch +
        index_head.to(tl.int64)*stride_Q_head
    }
    Q_block_ptr = tl.make_block_ptr(
        base= Q + qvk_offest ,
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order = (1,0)
    )
    K_block_ptr = tl.make_block_ptr(
        base = K +qvk_offest , 
        shape = (HEAD_DIM, SEQ_LEN),
        strides=(
            stride_K_dim, 
            stride_K_seq,
        ),
        offsets=(0,0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(1,0),
    )
    V_block_ptr = tl.make_block_ptr(
        base= V + qvk_offest, 
        strides=(stride_V_seq, stride_V_dim),
        offsets =(0,0) ,
        block_shape=(BLOCK_SIZE_KV,  HEAD_DIM),
        order=(0,1),
    )
    O_block_ptr = tl.make_block_ptr(
        base = Q + qvk_offest,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq,stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q , 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1,0)
    )

    offs_q= block_index_q * BLOCK_SIZE_Q + tl.arange(0,BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    m_i=  tl.zeros([BLOCK_SIZE_Q], dtype = tl.float32) - float("-inf")
    
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # acc: the accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptr)

    # Stage: 3 if causal, else 1

    if STAGE == 1 or STAGE == 3:
        # This step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    
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

