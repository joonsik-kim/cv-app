"""
AOI 솔더 페이스트 이미지 분석 스크립트

OMRON AOI 장비의 경사도 기반 False Color 이미지에서
솔더 페이스트 영역을 추출하기 위한 3가지 방법을 테스트합니다.

색상 인코딩:
- 🔴 빨강: 평탄한 부분 (낮은 경사)
- 🟢 녹색: 중간 경사 (측면 + 기판)
- 🔵 파랑: 급경사 (가장자리)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class SolderSegmentation:
    """솔더 페이스트 세그멘테이션 클래스"""

    def __init__(self, image_path):
        """
        Args:
            image_path: 이미지 파일 경로
        """
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(image_path))

        if self.image is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

        self.height, self.width = self.image.shape[:2]
        self.results = {}

    def method1_exclude_board(self, board_lower_hue=35, board_upper_hue=95,
                              board_lower_sat=30, board_upper_sat=255,
                              board_lower_val=30, board_upper_val=255):
        """
        방법 1: 기판(녹색 계열) 제외 → 나머지 = 솔더

        경사도 색상 인코딩:
        - 🔴 빨강: 평탄 (솔더 상면 + 기판 평탄면)
        - 🟢 녹색/청록: 중간 경사 (기판 영역)
        - 🔵 파랑: 급경사 (솔더 가장자리)

        전략: 기판의 진녹색~청록색 영역을 찾아서 제외

        Args:
            board_lower_hue: 기판 Hue 최소값 (기본 35)
            board_upper_hue: 기판 Hue 최대값 (기본 95)
            board_lower_sat: 기판 Saturation 최소값 (기본 30)
            board_upper_sat: 기판 Saturation 최대값 (기본 255)
            board_lower_val: 기판 Value 최소값 (기본 30)
            board_upper_val: 기판 Value 최대값 (기본 255)

        Returns:
            mask: 이진 마스크 (솔더=255, 기판=0)
        """
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # 기판 색상 범위 (녹색~청록)
        lower_board = np.array([board_lower_hue, board_lower_sat, board_lower_val])
        upper_board = np.array([board_upper_hue, board_upper_sat, board_upper_val])

        # 기판 마스크 → 반전 = 솔더
        board_mask = cv2.inRange(hsv, lower_board, upper_board)
        solder_mask = cv2.bitwise_not(board_mask)

        kernel = np.ones((3, 3), np.uint8)
        solder_mask = cv2.morphologyEx(solder_mask, cv2.MORPH_OPEN, kernel)
        solder_mask = cv2.morphologyEx(solder_mask, cv2.MORPH_CLOSE, kernel)

        self.results['method1_exclude_board'] = solder_mask
        return solder_mask

    def method2_hsv_blob(self):
        """
        방법 2: HSV Hue 채널 + Otsu 자동 이진화 + 최대 컨투어 ⭐ 추천

        HSV에서 기판(청록)과 솔더(비-청록)의 Hue 차이가 뚜렷함.
        Otsu로 자동 임계값 → 가장 큰 덩어리 = 솔더.

        Returns:
            mask: 이진 마스크 (솔더=255, 기판=0)
        """
        # GaussianBlur 전처리 → Hue 노이즈 감소, Otsu 안정성 향상
        blurred = cv2.GaussianBlur(self.image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]

        # Otsu 자동 이진화 (기판 Hue vs 솔더 Hue 자동 분리)
        _, binary = cv2.threshold(h_channel, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 가장 큰 컨투어만 남기기 = 솔더 덩어리
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(binary)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)

        self.results['method2_hsv_blob'] = mask
        self.results['method2_hue_channel'] = h_channel
        return mask

    def method3_kmeans(self, k=2):
        """
        방법 3: K-means 클러스터링으로 자동 분류

        픽셀을 BGR 좌표로 표현하여 k개 그룹으로 분류
        B 채널이 가장 높은 클러스터 = 솔더 (급경사 = 파랑)

        Args:
            k: 클러스터 개수 (기본 2: 솔더 vs 기판)

        Returns:
            mask: 이진 마스크
        """
        pixels = self.image.reshape((-1, 3)).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10,
                                         cv2.KMEANS_PP_CENTERS)

        # B - R 차이가 가장 큰 클러스터 = 솔더
        blue_dominance = centers[:, 0] - centers[:, 2]  # BGR: B=0, R=2
        solder_cluster = np.argmax(blue_dominance)

        mask = (labels.flatten() == solder_cluster).astype(np.uint8) * 255
        mask = mask.reshape((self.height, self.width))

        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self.results['method3_kmeans'] = mask
        return mask

    def calculate_area(self, mask, pixel_size_mm=0.01465):
        """
        마스크에서 솔더 면적 계산

        Args:
            mask: 이진 마스크
            pixel_size_mm: 1픽셀의 실제 크기 (mm) - AOI 스펙 기준

        Returns:
            dict: 면적 정보 (픽셀, mm²)
        """
        # 컨투어 추출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                'total_pixels': 0,
                'total_mm2': 0.0,
                'num_contours': 0
            }

        # 전체 픽셀 수 계산
        total_pixels = sum(cv2.contourArea(c) for c in contours)

        # mm² 변환
        total_mm2 = total_pixels * (pixel_size_mm ** 2)

        return {
            'total_pixels': int(total_pixels),
            'total_mm2': round(total_mm2, 6),
            'num_contours': len(contours)
        }

    def visualize_all(self, save_path=None):
        """
        3가지 방법의 결과를 비교 시각화

        Args:
            save_path: 저장 경로 (None이면 저장 안 함)
        """
        # Figure 생성 (2행 4열)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Solder Segmentation: {self.image_path.name}',
                     fontsize=14, fontweight='bold')

        # 원본 이미지 (BGR → RGB 변환)
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # HSV 변환
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        axes[0, 1].imshow(hsv)
        axes[0, 1].set_title('HSV Color Space')
        axes[0, 1].axis('off')

        # Hue 채널
        if 'method2_hue_channel' in self.results:
            axes[0, 2].imshow(self.results['method2_hue_channel'], cmap='hsv')
            axes[0, 2].set_title('Hue Channel')
            axes[0, 2].axis('off')

        # 빈 공간
        axes[0, 3].axis('off')

        # 방법 1: 기판 제외
        if 'method1_exclude_board' in self.results:
            mask1 = self.results['method1_exclude_board']
            area1 = self.calculate_area(mask1)
            axes[1, 0].imshow(mask1, cmap='gray')
            axes[1, 0].set_title(f'Method 1: Exclude Board\n'
                                 f'{area1["total_pixels"]} px | '
                                 f'{area1["total_mm2"]:.4f} mm²')
            axes[1, 0].axis('off')

        # 방법 2: HSV Blob
        if 'method2_hsv_blob' in self.results:
            mask2 = self.results['method2_hsv_blob']
            area2 = self.calculate_area(mask2)
            axes[1, 1].imshow(mask2, cmap='gray')
            axes[1, 1].set_title(f'Method 2: HSV Blob\n'
                                 f'{area2["total_pixels"]} px | '
                                 f'{area2["total_mm2"]:.4f} mm²')
            axes[1, 1].axis('off')

        # 방법 3: K-means
        if 'method3_kmeans' in self.results:
            mask3 = self.results['method3_kmeans']
            area3 = self.calculate_area(mask3)
            axes[1, 2].imshow(mask3, cmap='gray')
            axes[1, 2].set_title(f'Method 3: K-means\n'
                                 f'{area3["total_pixels"]} px | '
                                 f'{area3["total_mm2"]:.4f} mm²')
            axes[1, 2].axis('off')

        # 오버레이 (Method 2 기준 - 가장 적합)
        if 'method2_hsv_blob' in self.results:
            overlay = rgb_image.copy()
            mask = self.results['method2_hsv_blob']
            overlay[mask > 0] = [0, 255, 0]
            axes[1, 3].imshow(overlay)
            axes[1, 3].set_title('Method 2 Overlay')
            axes[1, 3].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 결과 저장: {save_path}")

        plt.show()


def analyze_all_images(image_dir, output_dir):
    """
    디렉토리 내 모든 이미지를 분석

    Args:
        image_dir: 이미지 디렉토리 경로
        output_dir: 결과 저장 디렉토리
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 파일 찾기
    image_files = sorted(image_dir.glob('*.png'))

    if not image_files:
        print(f"❌ 이미지 파일을 찾을 수 없음: {image_dir}")
        return

    print(f"📊 총 {len(image_files)}개 이미지 분석 시작\n")

    for img_path in image_files:
        print(f"🔍 분석 중: {img_path.name}")

        try:
            seg = SolderSegmentation(img_path)

            # 3가지 방법 실행
            seg.method1_exclude_board()
            seg.method2_hsv_blob()
            seg.method3_kmeans()

            # 결과 시각화 및 저장
            save_path = output_dir / f"result_{img_path.stem}.png"
            seg.visualize_all(save_path)

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            continue

    print(f"\n✅ 모든 분석 완료! 결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    # 경로 설정
    IMAGE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "images"
    OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

    # 전체 이미지 분석
    analyze_all_images(IMAGE_DIR, OUTPUT_DIR)
