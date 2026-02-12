"""
AOI 솔더 페이스트 이미지 분석 스크립트

이 스크립트는 Height Map (False Color) 이미지에서
솔더 페이스트 영역을 추출하기 위한 3가지 방법을 테스트합니다.
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

    def method1_hsv_color(self, lower_hue=0, upper_hue=30,
                          lower_sat=100, upper_sat=255,
                          lower_val=100, upper_val=255):
        """
        방법 1: HSV 색공간에서 빨강-오렌지 범위 추출

        빨강/오렌지 = 높은 부분 = 솔더 페이스트
        파랑/검정 = 낮은 부분 = 기판

        Args:
            lower_hue: Hue 최소값 (0-179)
            upper_hue: Hue 최대값 (0-179)
            lower_sat: Saturation 최소값 (0-255)
            upper_sat: Saturation 최대값 (0-255)
            lower_val: Value 최소값 (0-255)
            upper_val: Value 최대값 (0-255)

        Returns:
            mask: 이진 마스크 (솔더=255, 배경=0)
        """
        # BGR → HSV 변환
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # 빨강-오렌지 범위 정의
        lower = np.array([lower_hue, lower_sat, lower_val])
        upper = np.array([upper_hue, upper_sat, upper_val])

        # 마스크 생성
        mask = cv2.inRange(hsv, lower, upper)

        # 노이즈 제거 (Opening: 침식 → 팽창)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 구멍 메우기 (Closing: 팽창 → 침식)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self.results['method1_hsv'] = mask
        return mask

    def method2_channel_diff(self, threshold=30):
        """
        방법 2: R-B 채널 차이로 높이 분리

        높은 부분(솔더): R(빨강) 채널 값이 높음
        낮은 부분(기판): B(파랑) 채널 값이 높음
        → R - B 차이가 크면 솔더

        Args:
            threshold: 임계값 (기본 30)

        Returns:
            mask: 이진 마스크
        """
        # BGR 채널 분리
        b, g, r = cv2.split(self.image)

        # R - B 차이 계산
        diff = r.astype(np.int16) - b.astype(np.int16)
        diff = np.clip(diff, 0, 255).astype(np.uint8)

        # 임계값으로 이진화
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self.results['method2_channel_diff'] = mask
        self.results['method2_diff_image'] = diff
        return mask

    def method3_kmeans(self, k=2):
        """
        방법 3: K-means 클러스터링으로 자동 분류

        픽셀을 RGB 좌표로 표현하여 k개 그룹으로 분류
        가장 밝은(빨강 계열) 클러스터 = 솔더

        Args:
            k: 클러스터 개수 (기본 2: 솔더 vs 기판)

        Returns:
            mask: 이진 마스크
        """
        # 이미지를 1D 배열로 변환 (H*W, 3)
        pixels = self.image.reshape((-1, 3)).astype(np.float32)

        # K-means 클러스터링
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10,
                                         cv2.KMEANS_PP_CENTERS)

        # 클러스터 중심의 밝기(R 채널 값) 계산
        brightness = centers[:, 2]  # BGR이므로 R은 인덱스 2

        # 가장 밝은 클러스터 찾기
        brightest_cluster = np.argmax(brightness)

        # 해당 클러스터에 속하는 픽셀만 선택
        mask = (labels.flatten() == brightest_cluster).astype(np.uint8) * 255
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

        # R-B 차이 이미지
        if 'method2_diff_image' in self.results:
            axes[0, 2].imshow(self.results['method2_diff_image'], cmap='hot')
            axes[0, 2].set_title('R - B Channel Diff')
            axes[0, 2].axis('off')

        # 빈 공간
        axes[0, 3].axis('off')

        # 방법 1: HSV
        if 'method1_hsv' in self.results:
            mask1 = self.results['method1_hsv']
            area1 = self.calculate_area(mask1)
            axes[1, 0].imshow(mask1, cmap='gray')
            axes[1, 0].set_title(f'Method 1: HSV\n'
                                 f'{area1["total_pixels"]} px | '
                                 f'{area1["total_mm2"]:.4f} mm²')
            axes[1, 0].axis('off')

        # 방법 2: R-B
        if 'method2_channel_diff' in self.results:
            mask2 = self.results['method2_channel_diff']
            area2 = self.calculate_area(mask2)
            axes[1, 1].imshow(mask2, cmap='gray')
            axes[1, 1].set_title(f'Method 2: R-B Diff\n'
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

        # 오버레이 (가장 좋은 방법)
        if 'method1_hsv' in self.results:
            overlay = rgb_image.copy()
            mask = self.results['method1_hsv']
            overlay[mask > 0] = [0, 255, 0]  # 녹색
            axes[1, 3].imshow(overlay)
            axes[1, 3].set_title('Method 1 Overlay')
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
            seg.method1_hsv_color()
            seg.method2_channel_diff()
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
