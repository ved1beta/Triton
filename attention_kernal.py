import torch
import triton
import triton.language as tl

# Step 1: Basic matrix multiplication kernel to understand Triton fundamentals
@triton.jit
def basic_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr
):
    """Basic matrix multiplication kernel to understand Triton basics"""
    # Program ID gives us the block we're processing
    pid = tl.program_id(0)
    
    # Calculate row and column indices
    row_start = (pid // (N // BLOCK_SIZE)) * BLOCK_SIZE
    col_start = (pid % (N // BLOCK_SIZE)) * BLOCK_SIZE
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE):
        # Load blocks from A and B
        a = tl.load(a_ptr + row_start * stride_am + k * stride_ak)
        b = tl.load(b_ptr + k * stride_bk + col_start * stride_bn)
        
        # Perform matrix multiplication
        acc += tl.dot(a, b)
    
    # Store result
    c = tl.store(c_ptr + row_start * stride_cm + col_start * stride_cn, acc)

# Step 2: Query-Key multiplication kernel
@triton.jit
def attention_qk_kernel(
    q_ptr, k_ptr, output_ptr,
    seq_length, head_size,
    BLOCK_SIZE: tl.constexpr
):
    """Compute Query-Key multiplication for attention"""
    pid = tl.program_id(0)
    
    # Calculate row and column for this block
    row_idx = pid // (seq_length // BLOCK_SIZE)
    col_idx = pid % (seq_length // BLOCK_SIZE)
    
    # Load query and key blocks
    q_start = q_ptr + row_idx * BLOCK_SIZE * head_size
    k_start = k_ptr + col_idx * BLOCK_SIZE * head_size
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Compute attention scores
    for i in range(0, head_size, BLOCK_SIZE):
        q_block = tl.load(q_start + i)
        k_block = tl.load(k_start + i)
        acc += tl.dot(q_block, tl.trans(k_block))
    
    # Scale attention scores
    acc = acc / tl.sqrt(float(head_size))
    
    # Store result
    output_start = output_ptr + row_idx * BLOCK_SIZE * seq_length + col_idx * BLOCK_SIZE
    tl.store(output_start, acc)

# Step 3: Softmax kernel
@triton.jit
def softmax_kernel(
    ptr_in, ptr_out,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """Compute softmax operation"""
    row_idx = tl.program_id(0)
    row_start = ptr_in + row_idx * n_cols
    
    # Load input row
    row = tl.load(row_start + tl.arange(0, n_cols))
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Compute exponentials
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    
    # Compute softmax
    softmax_output = numerator / denominator
    
    # Store result
    tl.store(ptr_out + row_idx * n_cols + tl.arange(0, n_cols), softmax_output)

class TritonAttention(torch.nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        
    def forward(self, q, k, v):
        batch_size = q.shape[0]
        seq_len = q.shape[1]
        
        # Grid for parallel execution
        grid = (batch_size * seq_len * seq_len) // 256
        
        # Compute attention scores
        attention_scores = torch.empty(
            (batch_size, seq_len, seq_len),
            device='cuda',
            dtype=torch.float32
        )
        
        # Launch attention kernels
        attention_qk_kernel[grid](
            q.data_ptr(),
            k.data_ptr(),
            attention_scores.data_ptr(),
            seq_len,
            self.head_size,
            BLOCK_SIZE=16
        )
        
        # Apply softmax
        attention_probs = torch.empty_like(attention_scores)
        softmax_kernel[grid](
            attention_scores.data_ptr(),
            attention_probs.data_ptr(),
            seq_len,
            BLOCK_SIZE=256
        )
        
        # Compute final attention output
        output = torch.empty_like(q)
        attention_qk_kernel[grid](
            attention_probs.data_ptr(),
            v.data_ptr(),
            output.data_ptr(),
            seq_len,
            self.head_size,
            BLOCK_SIZE=16
        )
        
        return output

# Example usage and testing
def test_attention():
    head_size = 64
    seq_len = 256
    batch_size = 32
    
    # Create random input tensors
    q = torch.randn(batch_size, seq_len, head_size, device='cuda')
    k = torch.randn(batch_size, seq_len, head_size, device='cuda')
    v = torch.randn(batch_size, seq_len, head_size, device='cuda')
    
    # Initialize attention module
    attention = TritonAttention(head_size)
    
    # Forward pass
    output = attention(q, k, v)
    
    # Compare with PyTorch's native attention
    torch_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    # Check accuracy
    max_diff = torch.max(torch.abs(output - torch_output))
    print(f"Maximum difference from PyTorch implementation: {max_diff}")

if __name__ == "__main__":
    test_attention()