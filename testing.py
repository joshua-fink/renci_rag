import torch

# Create a tensor with size (1, 5, 1)
original_tensor = torch.tensor([[[1, 2, 3, 4, 5]]])

# Reduce dimension using torch.squeeze
squeezed_tensor = torch.squeeze(original_tensor)

print("Original Tensor:")
print(original_tensor)
print("Original Tensor Shape:", original_tensor.shape)

print("\nSqueezed Tensor:")
print(squeezed_tensor)
print("Squeezed Tensor Shape:", squeezed_tensor.shape)