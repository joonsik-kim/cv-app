"""
모폴로지 연산 테스트 - 실제 솔더 이미지에 적용

커널 사이즈별로 OPEN / CLOSE 결과를 비교한다.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from analyze_images import extract_board_color_from_og

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def get_raw_kmeans_mask(cropped_image, board_ref_lab, kmeans_k=3):
    """K-means까지만 실행 → 모폴로지 전 마스크 반환"""
    img_h, img_w = cropped_image.shape[:2]
    denoised = cv2.bilateralFilter(cropped_image, d=5, sigmaColor=50, sigmaSpace=50)

    lab_f32 = cv2.cvtColor(denoised, cv2.COLOR_BGR2Lab).astype(np.float32)
    pixels_lab = lab_f32.reshape(-1, 3)

    pos_weight = 10
    ys, xs = np.mgrid[0:img_h, 0:img_w]
    xs_norm = (xs.reshape(-1, 1).astype(np.float32) / max(img_w, 1)) * pos_weight
    ys_norm = (ys.reshape(-1, 1).astype(np.float32) / max(img_h, 1)) * pos_weight
    features = np.hstack([pixels_lab, xs_norm, ys_norm])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, _ = cv2.kmeans(
        features, kmeans_k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()

    bgr_flat = denoised.reshape(-1, 3).astype(np.float32)
    fillet_clusters = []
    for k_idx in range(kmeans_k):
        cluster_mask = labels == k_idx
        if np.sum(cluster_mask) < 30:
            continue
        mean_bgr = np.mean(bgr_flat[cluster_mask], axis=0)
        b_mean, _, r_mean = mean_bgr
        ratio = b_mean / max(r_mean, 1)
        if ratio > 0.8:
            fillet_clusters.append(k_idx)

    raw_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for k_idx in fillet_clusters:
        raw_mask[labels.reshape(img_h, img_w) == k_idx] = 255

    return raw_mask


if __name__ == '__main__':
    data_dir = Path(__file__).resolve().parent.parent.parent.parent / 'data' / '922R25-07830730'

    # 기판색 추출
    og_image = cv2.imread(str(data_dir / '922R25-07830730_001.png'))
    board_ref_lab, _ = extract_board_color_from_og(og_image)

    # 크롭 이미지 3장 테스트
    crop_paths = sorted(data_dir.glob('*_00[2-5].png'))[:3]

    for crop_path in crop_paths:
        cropped = cv2.imread(str(crop_path))
        raw_mask = get_raw_kmeans_mask(cropped, board_ref_lab)
        img_h, img_w = cropped.shape[:2]

        print(f"\n=== {crop_path.name} ({img_w}x{img_h}) ===")
        print(f"원본 흰색: {np.sum(raw_mask == 255)}px")

        kernel_sizes = [3, 5, 7, 9]
        fig, axes = plt.subplots(len(kernel_sizes) + 1, 4, figsize=(14, 3.5 * (len(kernel_sizes) + 1)))

        # 첫 행: 원본 + 크롭 이미지
        axes[0][0].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        axes[0][0].set_title('원본 크롭', fontsize=11)
        axes[0][1].imshow(raw_mask, cmap='gray', vmin=0, vmax=255)
        axes[0][1].set_title(f'K-means 마스크\n({np.sum(raw_mask == 255)}px)', fontsize=11)
        axes[0][2].axis('off')
        axes[0][3].axis('off')

        for i, ksize in enumerate(kernel_sizes):
            row = i + 1
            kernel = np.ones((ksize, ksize), np.uint8)

            opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
            both = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

            px_open = np.sum(opened == 255)
            px_both = np.sum(both == 255)

            axes[row][0].imshow(raw_mask, cmap='gray', vmin=0, vmax=255)
            axes[row][0].set_title(f'원본 마스크', fontsize=11)

            axes[row][1].imshow(opened, cmap='gray', vmin=0, vmax=255)
            axes[row][1].set_title(f'OPEN {ksize}x{ksize}\n({px_open}px)', fontsize=11)

            axes[row][2].imshow(both, cmap='gray', vmin=0, vmax=255)
            axes[row][2].set_title(f'OPEN→CLOSE {ksize}x{ksize}\n({px_both}px)', fontsize=11)

            # 오버레이: 원본 위에 마스크 표시
            overlay = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).copy()
            overlay[both > 0] = [255, 0, 0]  # 솔더 영역 빨간색
            axes[row][3].imshow(overlay)
            axes[row][3].set_title(f'원본에 오버레이\n(빨강=솔더)', fontsize=11)

            print(f"  {ksize}x{ksize}: OPEN {px_open}px → OPEN+CLOSE {px_both}px")

        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')

        plt.suptitle(f'{crop_path.name} ({img_w}x{img_h}) 모폴로지 비교', fontsize=13, fontweight='bold')
        plt.tight_layout()

        save_name = f'morphology_{crop_path.stem}.png'
        plt.savefig(f'outputs/{save_name}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  저장: outputs/{save_name}")
