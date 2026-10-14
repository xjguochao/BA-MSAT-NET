import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.ba_msat_net import BA_MSAT_Net
from dataloaders.cbct_dataset import CBCTSegDataset
from utils.losses import CompoundLoss


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducible training, aligned with paper experiment protocol."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_boundary_gt(labels: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Generate boundary ground truth from segmentation labels via 3D Sobel gradient operator.
    Implements the boundary-aware supervision described in BA-MSAT-Net.
    
    Args:
        labels: [B, D, H, W] integer tensor of class indices (4 classes)
        threshold: gradient magnitude threshold for binarization
    Returns:
        boundary_map: [B, 1, D, H, W] binary boundary mask
    """
    device = labels.device
    num_classes = 4
    
    # Convert labels to one-hot encoding: [B, C, D, H, W]
    one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=num_classes)
    one_hot = one_hot.permute(0, 4, 1, 2, 3).float()
    
    # 3D Sobel kernels for three axes
    sobel_d = torch.tensor([
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
    ], dtype=torch.float32, device=device)
    sobel_h = sobel_d.permute(2, 0, 1)
    sobel_w = sobel_d.permute(2, 1, 0)
    
    # Reshape for 3D convolution: [out_c, in_c, D, H, W]
    sobel_d = sobel_d.unsqueeze(0).unsqueeze(0)
    sobel_h = sobel_h.unsqueeze(0).unsqueeze(0)
    sobel_w = sobel_w.unsqueeze(0).unsqueeze(0)
    
    # Compute maximum gradient magnitude across all classes
    grad_magnitude = torch.zeros_like(one_hot[:, :1])
    for c in range(num_classes):
        class_map = one_hot[:, c:c+1]
        grad_d = torch.nn.functional.conv3d(class_map, sobel_d, padding=1)
        grad_h = torch.nn.functional.conv3d(class_map, sobel_h, padding=1)
        grad_w = torch.nn.functional.conv3d(class_map, sobel_w, padding=1)
        class_grad = torch.sqrt(grad_d ** 2 + grad_h ** 2 + grad_w ** 2)
        grad_magnitude = torch.max(grad_magnitude, class_grad)
    
    # Binarize to obtain boundary mask
    boundary_map = (grad_magnitude > threshold).float()
    return boundary_map


def main():
    parser = argparse.ArgumentParser(description="BA-MSAT-Net Training")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of CBCT image volumes')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory of segmentation label masks')
    parser.add_argument('--train_list', type=str, default=None, help='Text file with training filenames')
    parser.add_argument('--val_list', type=str, default=None, help='Text file with validation filenames')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training (paper: 2)')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate (paper: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (paper: 1e-4)')
    parser.add_argument('--gpu', type=str, default='0', help='CUDA device index')
    parser.add_argument('--log_dir', type=str, default='./logs/train_exp01', help='Directory for logs and checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Fix random seed
    set_seed(args.seed)

    # Device configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directories
    os.makedirs(os.path.join(args.log_dir, 'checkpoints'), exist_ok=True)

    # Dataset and dataloader
    train_ds = CBCTSegDataset(args.data_dir, args.label_dir, args.train_list, split='train')
    val_ds = CBCTSegDataset(args.data_dir, args.label_dir, args.val_list, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Model initialization (1 input channel, 4 output classes: crown, root, maxilla, mandible)
    model = BA_MSAT_Net(in_channels=1, num_classes=4).to(device)

    # Loss function (compound Dice + cross-entropy + boundary loss, paper formulation)
    criterion = CompoundLoss(lambda_sem=1.0, lambda_bnd=0.1)

    # Optimizer (Adam, matching paper hyperparameters)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # TensorBoard logger
    writer = SummaryWriter(args.log_dir)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Generate boundary ground truth
            boundary_gt = compute_boundary_gt(labels)

            # Forward pass
            seg_logits, boundary_logits = model(images)
            loss, sem_loss, bnd_loss = criterion(seg_logits, boundary_logits, labels, boundary_gt)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            # Log training progress
            if batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                print(f"Epoch {epoch+1:02d} | Batch {batch_idx:03d} | "
                      f"Loss: {loss.item():.4f} (Sem: {sem_loss.item():.4f}, Bnd: {bnd_loss.item():.4f})")
                writer.add_scalar('train/total_loss', loss.item(), global_step)
                writer.add_scalar('train/semantic_loss', sem_loss.item(), global_step)
                writer.add_scalar('train/boundary_loss', bnd_loss.item(), global_step)

        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                boundary_gt = compute_boundary_gt(labels)

                seg_logits, boundary_logits = model(images)
                loss, _, _ = criterion(seg_logits, boundary_logits, labels, boundary_gt)
                val_loss_total += loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        print(f"\nEpoch {epoch+1:02d} Summary | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")
        writer.add_scalar('val/loss', avg_val_loss, epoch)

        # Save best validation checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.log_dir, 'checkpoints', 'best_model.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"New best model saved at epoch {epoch+1} (val loss: {best_val_loss:.4f})\n")

    writer.close()
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
