import torch
import matplotlib.pyplot as plt

def plot_three_batches(tensor1, tensor2, tensor3, titles=('Tensor1', 'Tensor2', 'Tensor3'), num_images=3):
    """
    Plots images from three batches side-by-side in a compact layout.

    Args:
        tensor1, tensor2, tensor3 (torch.Tensor): 4D tensors of shape (B, C, H, W)
        titles (tuple): Titles for the 3 tensor types (column headers)
        num_images (int): Number of images to plot from each batch
    """
    assert tensor1.ndim == tensor2.ndim == tensor3.ndim == 4, "All tensors must be 4D (B, C, H, W)"

    tensor1 = tensor1[:num_images].detach().cpu()
    tensor2 = tensor2[:num_images].detach().cpu()
    tensor3 = tensor3[:num_images].detach().cpu()

    # Smaller per-image size: 2 inches wide, 2 inches tall
    fig, axes = plt.subplots(num_images, 3, figsize=(6, 2 * num_images))
    fig.canvas.manager.window.title("Virtual Try-On Dress System ")


    for i in range(num_images):
        for j, tensor in enumerate([tensor1, tensor2, tensor3]):
            image = tensor[i].permute(1, 2, 0).numpy()
            image = image.clip(0, 1)
            ax = axes[i, j] if num_images > 1 else axes[j]
            ax.imshow(image)
            ax.axis("off")
            ax.set_aspect('auto')  # Can change to 'equal' if needed
            if i == 0:
                ax.set_title(titles[j], fontsize=10)

    plt.subplots_adjust(wspace=0.05, hspace=0.1)  # tighter spacing
    plt.tight_layout()
    plt.show()
