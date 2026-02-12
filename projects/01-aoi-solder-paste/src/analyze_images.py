"""
AOI 솔더 필렛 계측 스크립트 (Hybrid 접근법)

OMRON AOI 경사도 기반 False Color 이미지에서 솔더 필렛 영역을 추출합니다.

Hybrid 전략:
1. OG(원본) 이미지에서 기판 배경색을 안정적으로 추출
2. Cropped(ROI) 이미지에 Color Distance 적용하여 필렛 영역 분리

색상 인코딩 (경사도 기반):
- 빨강: 평탄한 부분 (낮은 경사)
- 녹색: 중간 경사 (측면 + 기판)
- 파랑: 급경사 (가장자리)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_board_color_from_og(og_image):
    """
    OG 이미지에서 기판(PCB) 기준색을 추출한다.

    OG 이미지는 부품 전체 + 기판 배경이 보이므로,
    HSV Hue 범위로 녹색 기판 영역을 찾아 Lab 기준색을 계산한다.

    Args:
        og_image: BGR 이미지 (numpy array)

    Returns:
        board_ref_lab: 기판 기준색 (Lab, shape=(3,))
        board_mask: 기판 영역 마스크 (디버깅용)
    """
    hsv = cv2.cvtColor(og_image, cv2.COLOR_BGR2HSV)

    # 기판 = 녹색~청록 (Hue 35-95, Sat > 30, Val > 20)
    board_mask = cv2.inRange(
        hsv,
        np.array([35, 30, 20]),
        np.array([95, 255, 255])
    )

    # 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_OPEN, kernel)

    # Lab 변환 후 기판 영역 픽셀만 추출
    lab = cv2.cvtColor(og_image, cv2.COLOR_BGR2Lab)
    board_pixels = lab[board_mask > 0].astype(np.float64)

    if len(board_pixels) < 10:
        # 기판 검출 실패 시 fallback: 이미지 네 모서리에서 추출
        h, w = og_image.shape[:2]
        corners = np.vstack([
            lab[0:5, 0:5].reshape(-1, 3),
            lab[0:5, -5:].reshape(-1, 3),
            lab[-5:, 0:5].reshape(-1, 3),
            lab[-5:, -5:].reshape(-1, 3),
        ]).astype(np.float64)
        board_ref_lab = np.median(corners, axis=0)
    else:
        board_ref_lab = np.median(board_pixels, axis=0)

    return board_ref_lab, board_mask


def measure_fillet(cropped_image, board_ref_lab):
    """
    Cropped(ROI) 이미지에서 솔더 필렛 영역을 추출하고 면적을 계측한다.

    3단계 파이프라인:
    1. 기판 제거: OG 기준색과의 Lab Color Distance → Otsu 이진화
    2. 바디 제거: 어두운 픽셀 (V < threshold) 제거
    3. 필렛 분리: 파란색(급경사=필렛 엣지) 기반으로 필렛 영역만 추출
       - 파란색 픽셀 찾기 → 팽창 → 인접 주황/빨강도 포함
       - 랜드는 파란색이 없으므로 자동 제외

    Args:
        cropped_image: BGR 이미지 (numpy array)
        board_ref_lab: OG에서 추출한 기판 기준색 (Lab)

    Returns:
        dict: {
            'mask': 이진 마스크 (필렛=255, 나머지=0),
            'distance_map': 정규화된 거리맵 (0-255),
            'blue_mask': 파란색(급경사) 검출 마스크 (디버깅용),
            'area_pixels': 필렛 면적 (픽셀),
            'area_mm2': 필렛 면적 (mm²),
        }
    """
    PIXEL_SIZE_MM = 0.01465

    # === Stage 1: 기판 제거 (Lab Color Distance) ===
    lab = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2Lab).astype(np.float64)
    diff = lab - board_ref_lab
    distance = np.sqrt(np.sum(diff ** 2, axis=2)).astype(np.float32)
    dist_norm = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, non_board = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # === Stage 2: 바디 제거 (어두운 영역) ===
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    not_dark = (v >= 40).astype(np.uint8) * 255

    # === Stage 3: 필렛 분리 (파란색 기반) ===
    # 파란색 = 급경사 = 필렛의 고유 특징 (Hue 85-130)
    blue_mask = cv2.inRange(hsv, np.array([85, 20, 40]), np.array([130, 255, 255]))

    # 파란색 영역을 팽창시켜 인접한 필렛 표면(주황/빨강)까지 포함
    dilate_kernel = np.ones((7, 7), np.uint8)
    blue_expanded = cv2.dilate(blue_mask, dilate_kernel, iterations=2)

    # 최종 필렛 마스크 = 기판아님 AND 바디아님 AND (파란색 OR 파란색 근처)
    fillet_mask = cv2.bitwise_and(non_board, not_dark)
    fillet_mask = cv2.bitwise_and(fillet_mask, blue_expanded)

    # 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    fillet_mask = cv2.morphologyEx(fillet_mask, cv2.MORPH_OPEN, kernel)
    fillet_mask = cv2.morphologyEx(fillet_mask, cv2.MORPH_CLOSE, kernel)

    # 면적 계산
    contours, _ = cv2.findContours(fillet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_pixels = 0
    clean_mask = np.zeros_like(fillet_mask)

    if contours:
        min_area = max(3, fillet_mask.size * 0.003)
        for c in contours:
            a = cv2.contourArea(c)
            if a >= min_area:
                cv2.drawContours(clean_mask, [c], -1, 255, cv2.FILLED)
                area_pixels += int(a)

    area_mm2 = area_pixels * (PIXEL_SIZE_MM ** 2)

    return {
        'mask': clean_mask,
        'distance_map': dist_norm,
        'blue_mask': blue_mask,
        'area_pixels': area_pixels,
        'area_mm2': round(area_mm2, 6),
    }


def visualize_part_results(part_name, og_image, board_mask, board_ref_lab,
                           cropped_images, results, save_path=None):
    """
    한 부품의 모든 ROI 결과를 시각화한다.

    1행: OG 원본 | OG 기판마스크 | (빈칸 또는 요약)
    2행~: 각 Cropped의 원본 | Distance Map | 필렛 마스크 | 오버레이

    Args:
        part_name: 부품번호
        og_image: OG BGR 이미지
        board_mask: 기판 검출 마스크
        board_ref_lab: 기판 기준색
        cropped_images: [(filename, BGR image), ...]
        results: [measure_fillet 결과 dict, ...]
        save_path: 저장 경로
    """
    n_crops = len(cropped_images)
    n_rows = 1 + n_crops
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 3 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Hybrid Fillet Measurement: {part_name}', fontsize=13, fontweight='bold')

    # --- Row 0: OG 이미지 ---
    og_rgb = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(og_rgb)
    axes[0, 0].set_title('OG Original')
    axes[0, 0].axis('off')

    # 기판 마스크 오버레이
    og_overlay = og_rgb.copy()
    og_overlay[board_mask > 0] = [0, 255, 0]
    axes[0, 1].imshow(og_overlay)
    axes[0, 1].set_title(f'Board Detection\nRef Lab: [{board_ref_lab[0]:.0f}, {board_ref_lab[1]:.0f}, {board_ref_lab[2]:.0f}]')
    axes[0, 1].axis('off')

    # 면적 요약
    total_px = sum(r['area_pixels'] for r in results)
    total_mm2 = sum(r['area_mm2'] for r in results)
    summary_text = f"Total Fillet Area\n{n_crops} ROIs\n{total_px} px\n{total_mm2:.4f} mm²"
    axes[0, 2].text(0.5, 0.5, summary_text, ha='center', va='center',
                    fontsize=11, transform=axes[0, 2].transAxes)
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')
    axes[0, 4].axis('off')

    # --- Row 1+: 각 Cropped 결과 ---
    for i, ((fname, crop_img), result) in enumerate(zip(cropped_images, results)):
        row = i + 1
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

        # 원본
        axes[row, 0].imshow(crop_rgb)
        axes[row, 0].set_title(f'{fname}', fontsize=9)
        axes[row, 0].axis('off')

        # Distance Map (기판과의 거리)
        axes[row, 1].imshow(result['distance_map'], cmap='hot')
        axes[row, 1].set_title('Board Distance', fontsize=9)
        axes[row, 1].axis('off')

        # Blue mask (급경사 = 필렛 엣지)
        axes[row, 2].imshow(result['blue_mask'], cmap='Blues')
        axes[row, 2].set_title('Blue (steep)', fontsize=9)
        axes[row, 2].axis('off')

        # 필렛 마스크
        axes[row, 3].imshow(result['mask'], cmap='gray')
        axes[row, 3].set_title(f"Fillet: {result['area_pixels']} px\n{result['area_mm2']:.4f} mm²",
                               fontsize=9)
        axes[row, 3].axis('off')

        # 오버레이
        overlay = crop_rgb.copy()
        overlay[result['mask'] > 0] = [0, 255, 0]
        axes[row, 4].imshow(overlay)
        axes[row, 4].set_title('Overlay', fontsize=9)
        axes[row, 4].axis('off')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  -> 결과 저장: {save_path}")

    plt.show()


def analyze_part(part_name, data_dir, output_dir):
    """
    한 부품 타입의 모든 이미지를 hybrid 방식으로 분석한다.

    1. OG 이미지에서 기판 기준색 추출 (첫 번째 OG 사용)
    2. 모든 Cropped 이미지에 Color Distance 적용
    3. 결과 시각화 및 저장

    Args:
        part_name: 부품번호 (폴더명)
        data_dir: data/imgae_processing 경로
        output_dir: 결과 저장 디렉토리
    """
    og_dir = Path(data_dir) / "OG" / part_name
    crop_dir = Path(data_dir) / "Cropped" / part_name

    if not og_dir.exists():
        print(f"  OG 폴더 없음: {og_dir}")
        return
    if not crop_dir.exists():
        print(f"  Cropped 폴더 없음: {crop_dir}")
        return

    og_files = sorted(og_dir.glob("*.png"))
    crop_files = sorted(crop_dir.glob("*.png"))

    if not og_files or not crop_files:
        print(f"  이미지 없음: OG={len(og_files)}, Cropped={len(crop_files)}")
        return

    print(f"\n{'='*60}")
    print(f"부품: {part_name}")
    print(f"OG: {len(og_files)}장, Cropped: {len(crop_files)}장")
    print(f"{'='*60}")

    # --- Step 1: OG에서 기판 기준색 추출 ---
    # 모든 OG 이미지의 기판색 평균을 사용 (안정성 향상)
    all_board_refs = []
    first_og = None
    first_board_mask = None

    for og_path in og_files:
        og_img = cv2.imread(str(og_path))
        if og_img is None:
            continue
        ref, mask = extract_board_color_from_og(og_img)
        all_board_refs.append(ref)
        if first_og is None:
            first_og = og_img
            first_board_mask = mask

    if not all_board_refs:
        print("  OG 이미지 로드 실패")
        return

    # 모든 OG의 기판색 중앙값 = 최종 기준색
    board_ref_lab = np.median(np.array(all_board_refs), axis=0)
    print(f"  기판 기준색 (Lab): [{board_ref_lab[0]:.1f}, {board_ref_lab[1]:.1f}, {board_ref_lab[2]:.1f}]")
    print(f"  (OG {len(all_board_refs)}장에서 추출)")

    # --- Step 2: 각 Cropped에 Color Distance 적용 ---
    cropped_images = []
    results = []

    for crop_path in crop_files:
        crop_img = cv2.imread(str(crop_path))
        if crop_img is None:
            print(f"  Cropped 로드 실패: {crop_path.name}")
            continue

        result = measure_fillet(crop_img, board_ref_lab)
        cropped_images.append((crop_path.name, crop_img))
        results.append(result)

        print(f"  {crop_path.name}: {result['area_pixels']:>5d} px | {result['area_mm2']:.4f} mm²")

    # --- Step 3: 시각화 ---
    if results:
        save_path = Path(output_dir) / f"result_{part_name}.png"
        visualize_part_results(
            part_name, first_og, first_board_mask, board_ref_lab,
            cropped_images, results, save_path
        )

        # 요약 통계
        areas = [r['area_pixels'] for r in results]
        print(f"\n  --- 요약 ---")
        print(f"  평균: {np.mean(areas):.1f} px")
        print(f"  표준편차: {np.std(areas):.1f} px")
        print(f"  최소/최대: {min(areas)} / {max(areas)} px")


if __name__ == "__main__":
    # 경로 설정
    DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "imgae_processing"
    OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

    # 테스트 대상 부품
    TARGET_PART = "922101-08080730"

    analyze_part(TARGET_PART, DATA_DIR, OUTPUT_DIR)
