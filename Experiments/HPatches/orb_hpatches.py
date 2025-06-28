import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

MAX_KEYPOINT = 1000
OUTPUT_METHOD = 'orb'

def load_and_resize(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    original_size = img.shape[::-1]  # (w, h)
    resized = cv2.resize(img, (320, 240))
    return resized, original_size

def match_descriptors(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return sorted(good_matches, key=lambda x: x.distance)


def compute_matching_accuracy(matches, kp1, kp2, H, threshold=3):
    correct = 0
    for m in matches:
        pt1 = np.array([*kp1[m.queryIdx].pt, 1.0])
        projected = H @ pt1
        projected /= projected[2]
        pt2 = np.array(kp2[m.trainIdx].pt)
        error = np.linalg.norm(projected[:2] - pt2)
        if error < threshold:
            correct += 1
    return correct / len(matches) if matches else 0

def compute_mean_accuracy(values):
    return sum(values) / len(values) if values else 0

def plot_histogram(illum_results, view_results, save_path):
    illum_max = max(illum_results) if illum_results else 0
    illum_min = min(illum_results) if illum_results else 0
    illum_mean = compute_mean_accuracy(illum_results)

    view_max = max(view_results) if view_results else 0
    view_min = min(view_results) if view_results else 0
    view_mean = compute_mean_accuracy(view_results)

    labels = ['Illum Max', 'Illum Min', 'Illum Mean', 'View Max', 'View Min', 'View Mean']
    values = [illum_max, illum_min, illum_mean, view_max, view_min, view_mean]
    colors = ['orange']*3 + ['blue']*3

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.ylim(0, 1)
    plt.ylabel('Matching Accuracy')
    plt.title('HPatches Matching Accuracy Statistics (ORB)')
    plt.grid(axis='y')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                 f'{height*100:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    hpatches_root = '/home/lab605/lab605/dataset/HPatches'
    output_root = '/home/lab605/lab605/GCNv2_SLAM/ORB_matching'
    os.makedirs(output_root, exist_ok=True)

    orb = cv2.ORB_create(nfeatures=MAX_KEYPOINT)

    illumination_results = []
    viewpoint_results = []
    results = []

    seqs = sorted(glob(os.path.join(hpatches_root, '*')))
    for seq in seqs:
        name = os.path.basename(seq)

        for idx in range(1, 7):
            img_path = os.path.join(seq, f"{idx}.ppm")
            if not os.path.exists(img_path):
                continue

            img_resized, original_size = load_and_resize(img_path)
            kp, desc = orb.detectAndCompute(img_resized, None)
            if desc is None or len(kp) == 0:
                continue

            sx = original_size[0] / 320
            sy = original_size[1] / 240
            kpts_scaled = np.array([[pt.pt[0] * sx, pt.pt[1] * sy] for pt in kp])

            save_dir = os.path.join(hpatches_root, name, f"{idx}.ppm.{OUTPUT_METHOD}")
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "keypoints.npy"), kpts_scaled)
            np.save(os.path.join(save_dir, "descriptors.npy"), desc)

        # --- accuracy check on 1 vs 6 ---
        img1_path = os.path.join(seq, '1.ppm')
        img2_path = os.path.join(seq, '6.ppm')
        H_path = os.path.join(seq, 'H_1_6')

        if not all(os.path.exists(p) for p in [img1_path, img2_path, H_path]):
            continue

        img1, size1 = load_and_resize(img1_path)
        img2, size2 = load_and_resize(img2_path)

        kp1, desc1 = orb.detectAndCompute(img1, None)
        kp2, desc2 = orb.detectAndCompute(img2, None)
        if desc1 is None or desc2 is None:
            continue

        matches = match_descriptors(desc1, desc2)

        sx = 320 / size1[0]
        sy = 240 / size1[1]
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        H = np.loadtxt(H_path)
        H = S @ H @ np.linalg.inv(S)

        acc = compute_matching_accuracy(matches, kp1, kp2, H)

        if name.startswith('i'):
            illumination_results.append(acc)
        elif name.startswith('v'):
            viewpoint_results.append(acc)

        results.append((name, acc))
        print(f"{name}: Matching Accuracy = {acc:.4f}")

    with open(os.path.join(output_root, "matching_results.txt"), 'w') as f:
        for name, acc in results:
            f.write(f"{name}: {acc:.4f}\n")

    print("\n📊 ORB Feature Matching Summary:")
    print(f"Illumination Avg: {compute_mean_accuracy(illumination_results):.4f}")
    print(f"Viewpoint Avg:    {compute_mean_accuracy(viewpoint_results):.4f}")

    plot_histogram(
        illumination_results,
        viewpoint_results,
        os.path.join(output_root, 'orb_matching_accuracy_histogram.png')
    )

    print("✅ ORB evaluation finished!")
