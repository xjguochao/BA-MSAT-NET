import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.ba_msat_net import BA_MSAT_Net
from dataloaders.cbct_dataset import CBCTSegDataset
from utils.losses import CompoundLoss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of CBCT volumes')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory of label masks')
    parser.add_argument('--train_list', type=str, default=None, help='Text file listing training filenames')
    parser.add_argument('--val_list', type=str, default=None, help='Text file listing validation filenames')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--log_dir', type=str, default='./logs')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    train_ds = CBCTSegDataset(args.data_dir, args.label_dir, args.train_list, split='train')
    val_ds = CBCTSegDataset(args.data_dir, args.label_dir, args.val_list, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # Model
    model = BA_MSAT_Net(in_channels=1, num_classes=4).to(device)

    # Loss and optimizer
    criterion = CompoundLoss(lambda_sem=1.0, lambda_bnd=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Tensorboard
    writer = SummaryWriter(args.log_dir)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Generate boundary ground truth (simplified: use gradient magnitude or morphological boundary)
            # For demo, we use zeros; in practice compute boundary from labels.
            boundary_gt = torch.zeros_like(labels, dtype=torch.float32).unsqueeze(1)  # placeholder

            # Forward
            seg_logits, boundary_logits = model(images)

            # Loss
            loss, sem_loss, bnd_loss = criterion(seg_logits, boundary_logits, labels, boundary_gt)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f} (Sem: {sem_loss.item():.4f}, Bnd: {bnd_loss.item():.4f})")
                writer.add_scalar('train/loss', loss.item(), epoch*len(train_loader)+i)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                boundary_gt = torch.zeros_like(labels, dtype=torch.float32).unsqueeze(1)
                seg_logits, boundary_logits = model(images)
                loss, _, _ = criterion(seg_logits, boundary_logits, labels, boundary_gt)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")
        writer.add_scalar('val/loss', val_loss, epoch)

        # Save checkpoint
        torch.save(model.state_dict(), f"ba_msat_net_epoch{epoch+1}.pth")

    writer.close()

if __name__ == '__main__':
    main()