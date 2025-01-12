import torch
import triton 
import triton.language as tl
DEVICE = "cuda"

@triton.jit
def add_kernal(x_ptr, y_ptr, output_ptr , n_elements , BLOCK_SIZE: tl.constexpr, ):
    pid = tl.program_id(axis=0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start  + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x_ptr +offsets , mask=mask)

    y = tl.load(y_ptr +offsets , mask=mask)

    output = x + y
    tl.store(output_ptr + offsets , output , mask=mask)

def add(x:torch.Tensor , y : torch.Tensor ):
    output = torch.empty_like(x)
    n_elem = output.numel()

    x = x.to(DEVICE)
    y = y.to(DEVICE)
    output = torch.empty_like(x, device=DEVICE)



    grid = lambda meta : (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)

    add_kernal[grid](x, y , output, n_elem, BLOCK_SIZE= 1024)
    return output

size = 98432
x = torch.randn(size)
y = torch.randn(size)

output_torch = x+y
output_triton = add(x, y)
print(output_torch)

print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')