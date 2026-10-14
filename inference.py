import os
import argparse
import numpy as np
import torch
import nibabel as nib

from models.ba_msat_net import BA_MSAT_Net


def load_model(weights_path: str, device: str = 'cuda') -> BA_MSAT_Net:
    """Load trained BA-MSAT-Net model checkpoint."""
    model = BA_MSAT_Net(in_channels=1, num_classes=4)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def predict_volume(
    model: BA_MSAT_Net,
    volume_path: str,
    patch_size: tuple = (160, 160, 160),
    stride: tuple = (128, 128, 128),
    device: str = 'cuda'
) -> np.ndarray:
    """
    Sliding window inference for full CBCT volume, aligned with paper protocol.
    Patch size: 160×160×160 voxels (paper specification).
    Overlapping patches are averaged to reduce boundary artifacts.
    
    Args:
        model: trained BA-MSAT-Net model
        volume_path: path to input NIfTI CBCT volume
        patch_size: input patch size for inference (paper: 160×160×160)
        stride: sliding window stride
        device: computation device
    Returns:
        segmentation: [D, H, W] uint8 segmentation mask (0-3 classes)
    """
    # Load volume and preserve original spatial metadata
    img_obj = nib.load(volume_path)
    img = img_obj.get_fdata().astype(np.float32)
    D, H, W = img.shape

    # CBCT value windowing and normalization (paper preprocessing)
    img = np.clip(img, 500, 2500)
    img = (img - 500) / (2500 - 500)

    # Pad volume to fit patch sliding
    pad_d = max(patch_size[0] - D, 0)
    pad_h = max(patch_size[1] - H, 0)
    pad_w = max(patch_size[2] - W, 0)
    img_padded = np.pad(img, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

    # Initialize output and overlap counter
    output = np.zeros(img_padded.shape, dtype=np.float32)
    count = np.zeros(img_padded.shape, dtype=np.float32)

    # Sliding window inference
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    for d in range(0, img_padded.shape[0] - pd + 1, sd):
        for h in range(0, img_padded.shape[1] - ph + 1, sh):
            for w in range(0, img_padded.shape[2] - pw + 1, sw):
                # Extract patch and add batch/channel dimensions
                patch = img_padded[d:d+pd, h:h+ph, w:w+pw]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)

                # Forward pass
                with torch.no_grad():
                    seg_logits, _ = model(patch_tensor)
                    seg_pred = torch.argmax(seg_logits, dim=1).squeeze().cpu().numpy()

                # Accumulate prediction
                output[d:d+pd, h:h+ph, w:w+pw] += seg_pred
                count[d:d+pd, h:h+ph, w:w+pw] += 1

    # Average overlapping regions
    output = output / count
    segmentation = np.round(output).astype(np.uint8)

    # Remove padding, restore original volume size
    segmentation = segmentation[:D, :H, :W]
    return segmentation, img_obj.affine


def main():
    parser = argparse.ArgumentParser(description="BA-MSAT-Net Inference")
    parser.add_argument('--weights', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input CBCT NIfTI file')
    parser.add_argument('--output', type=str, required=True, help='Path to save output segmentation NIfTI')
    parser.add_argument('--gpu', type=str, default='0', help='CUDA device index')
    args = parser.parse_args()

    # Device configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and run inference
    model = load_model(args.weights, device)
    seg_mask, affine = predict_volume(model, args.input, device=device)

    # Save result with original spatial affine
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    nifti_out = nib.Nifti1Image(seg_mask, affine)
    nib.save(nifti_out, args.output)
    print(f"Segmentation saved to {args.output}")


if __name__ == '__main__':
    main()
