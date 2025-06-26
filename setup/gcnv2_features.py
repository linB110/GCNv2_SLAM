import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def load_and_preprocess(img_path):
    img_uint8 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img_uint8, (320, 240))
    img = img_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    return img_resized, tensor

def extract_features(model, img_tensor, max_keypoints=1000):  ## modified here if you want more or less keypoints
    with torch.no_grad():
        keypoints_tensor, desc = model(img_tensor)

        if keypoints_tensor.shape[0] == 0:
            return [], [], None

        keypoints_tensor = keypoints_tensor[:max_keypoints]
        desc = desc[:max_keypoints]

        pts = keypoints_tensor[:, :2].cpu().numpy()
        desc = desc.cpu().numpy()

        keypoints = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in pts]
        return keypoints, desc, None

def match_descriptors(desc1, desc2):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.match(desc1.astype(np.float32), desc2.astype(np.float32))
    return sorted(matches, key=lambda x: x.distance)

def draw_and_save_matches(img1, kp1, img2, kp2, matches, out_path, top_k=50):
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:top_k], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(out_path, match_img)

    # show image one after one
    #img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
    #plt.imshow(img_rgb)
    #plt.title(os.path.basename(out_path))
    #plt.axis('off')
    #plt.show()

def compute_matching_accuracy(matches, kp1, kp2, H, threshold=3):
    correct = 0
    total = len(matches)
    for m in matches:
        pt1 = np.array([*kp1[m.queryIdx].pt, 1.0])
        projected = H @ pt1
        projected /= projected[2]
        pt2 = np.array(kp2[m.trainIdx].pt)
        error = np.linalg.norm(projected[:2] - pt2)
        if error < threshold:
            correct += 1
    return correct / total if total > 0 else 0

# -------- main --------
if __name__ == '__main__':
    
    #device = torch.device('cpu')  # cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = '/home/lab605/lab605/GCNv2_SLAM/GCN2/gcn2_320x240.pt'
    #model = torch.jit.load(model_path, map_location=device)  #cpu
    model = torch.jit.load(model_path, map_location=device).to(device)

    model.eval()

    hpatches_root = '/home/lab605/lab605/dataset/HPatches'
    output_root = '/home/lab605/lab605/GCNv2_SLAM/GCN_matching'
    os.makedirs(output_root, exist_ok=True)

    results = []
    seqs = sorted(glob(os.path.join(hpatches_root, 'i_*')))
    for seq in seqs:
        name = os.path.basename(seq)
        for i in range(2, 7):
            img1_path = os.path.join(seq, '1.ppm')
            img2_path = os.path.join(seq, f'{i}.ppm')
            H_path = os.path.join(seq, f'H_1_{i}')

            if not all(os.path.exists(p) for p in [img1_path, img2_path, H_path]):
                print(f"❌ lack of {name} H_1_{i}, omitting")
                continue

            img1, tensor1 = load_and_preprocess(img1_path)
            img2, tensor2 = load_and_preprocess(img2_path)

            try:
                kp1, desc1, _ = extract_features(model, tensor1)
                kp2, desc2, _ = extract_features(model, tensor2)
            except Exception as e:
                print(f"❌ extract feature failed {name} 1-{i}: {e}")
                continue

            if len(kp1) == 0 or len(kp2) == 0:
                print(f"⚠️ can not detect feature point {name} 1-{i}, skipping")
                continue

            matches = match_descriptors(desc1, desc2)
            H = np.loadtxt(H_path)
            acc = compute_matching_accuracy(matches, kp1, kp2, H)

            print(f"{name} 1-{i}: Matching Accuracy = {acc:.4f}")
            results.append((f"{name}_1_{i}", acc))

            out_path = os.path.join(output_root, f"{name}_1_{i}_matches.png")
            draw_and_save_matches(img1, kp1, img2, kp2, matches, out_path)

    # store results
    with open(os.path.join(output_root, "matching_results.txt"), 'w') as f:
        for name, acc in results:
            f.write(f"{name}: {acc:.4f}\n")

    print("✅ task finished")

