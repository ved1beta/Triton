import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from nn import TritonNeuralNetwork, train_and_evaluate  # assuming previous code is saved as triton_nn.py

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("This implementation requires a CUDA-enabled GPU")
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load MNIST dataset
    print("Downloading and loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', 
                                 train=True, 
                                 download=True, 
                                 transform=transform)
    
    test_dataset = datasets.MNIST('./data', 
                                train=False, 
                                transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, 
                            batch_size=64, 
                            shuffle=True)
    
    test_loader = DataLoader(test_dataset, 
                           batch_size=64, 
                           shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    model = TritonNeuralNetwork(
        in_features=28*28,    # MNIST images are 28x28
        hidden_features=256,
        out_features=10       # 10 digits to classify
    ).cuda()
    
    # Train and evaluate the model
    print("Starting training...")
    metrics = train_and_evaluate(
        model,
        train_loader,
        test_loader,
        epochs=10,
        learning_rate=0.001
    )
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['test_loss'], label='Test Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['test_acc'], label='Test Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    print("Training results plot saved as 'training_results.png'")
    
    # Save the trained model
    torch.save(model.state_dict(), 'triton_nn_mnist.pth')
    print("Model saved as 'triton_nn_mnist.pth'")

if __name__ == "__main__":
    main()