# nn.py
import triton
import triton.language as tl
import torch
import torch.nn.functional as F

@triton.jit
def triton_nn_kernel(
    # Pointers to matrices
    input_ptr,
    weight_ptr,
    output_ptr,
    # Matrix dimensions
    batch_size,
    in_features,
    out_features,
    # Block size parameter
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Calculate indices
    batch_idx = pid // out_features
    out_idx = pid % out_features

    # Create offsets for the current block
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator
    acc = 0.0
    
    # Compute input and weight offsets
    input_block_ptr = input_ptr + batch_idx * in_features
    weight_block_ptr = weight_ptr + out_idx * in_features
    
    # Iterate over blocks
    for block_start in range(0, in_features, BLOCK_SIZE):
        # Create block mask
        block_mask = block_start + offsets < in_features
        
        # Load blocks using pointer arithmetic
        input_block = tl.load(input_block_ptr + block_start + offsets, mask=block_mask, other=0.0)
        weight_block = tl.load(weight_block_ptr + block_start + offsets, mask=block_mask, other=0.0)
        
        # Compute partial dot product
        acc += tl.sum(input_block * weight_block * block_mask, axis=0)
    
    # Write output
    output_offset = batch_idx * out_features + out_idx
    tl.store(output_ptr + output_offset, acc)

class TritonNeuralNetwork(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_features, hidden_features)
        self.layer2 = torch.nn.Linear(hidden_features, out_features)
        self.block_size = 32  # Can be tuned for performance

    def forward(self, x):
        # Flatten input if necessary
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        # First layer using Triton kernel
        batch_size = x.shape[0]
        output = torch.empty((batch_size, self.layer1.out_features), device=x.device)
        
        # Launch kernel
        grid = (batch_size * self.layer1.out_features,)
        triton_nn_kernel[grid](
            x.contiguous(),
            self.layer1.weight.contiguous(),
            output,
            batch_size,
            self.layer1.in_features,
            self.layer1.out_features,
            self.block_size,
        )
        
        # Apply activation and second layer
        hidden = F.relu(output)
        return self.layer2(hidden)

def train_and_evaluate(model, train_loader, test_loader, epochs=10, learning_rate=0.01):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        
        # Testing
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        test_loss = test_loss / len(test_loader)
        
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
    
    return metrics