import os
import argparse
import torch
import nibabel as nib
import numpy as np
from models.ba_msat_net import BA_MSAT_Net
from dataloaders.cbct_dataset import CBCTSegDataset

def load_model(weights_path, device='cuda'):
    model = BA_MSAT_Net(in_channels=1, num_classes=4)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_volume(model, volume_path, patch_size=(160,160,160), stride=(128,128,128), device='cuda'):
    # Load full volume
    img = nib.load(volume_path).get_fdata().astype(np.float32)
    # Normalize
    img = np.clip(img, 500, 2500)
    img = (img - 500) / (2500 - 500)

    D, H, W = img.shape
    # Pad to multiple of stride if needed
    pad_d = max(patch_size[0] - D, 0)
    pad_h = max(patch_size[1] - H, 0)
    pad_w = max(patch_size[2] - W, 0)
    img_padded = np.pad(img, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

    # Sliding window inference
    output = np.zeros(img_padded.shape, dtype=np.uint8)
    count = np.zeros(img_padded.shape, dtype=np.float32)

    for d in range(0, img_padded.shape[0] - patch_size[0] + 1, stride[0]):
        for h in range(0, img_padded.shape[1] - patch_size[1] + 1, stride[1]):
            for w in range(0, img_padded.shape[2] - patch_size[2] + 1, stride[2]):
                patch = img_padded[d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]]
                patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    seg_logits, _ = model(patch_t)
                    seg = torch.argmax(seg_logits, dim=1).squeeze().cpu().numpy()
                output[d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += seg
                count[d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += 1

    # Average overlapping predictions
    output = output / count
    output = np.round(output).astype(np.uint8)

    # Remove padding
    output = output[:D, :H, :W]
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.weights, device)
    seg = predict_volume(model, args.input, device=device)

    # Save result
    nifti_img = nib.Nifti1Image(seg, affine=np.eye(4))
    nib.save(nifti_img, args.output)

if __name__ == '__main__':
    main()