# 01. AOI 솔더 페이스트 계측 프로젝트

> OMRON AOI 장비의 Height Map 이미지에서 솔더 페이스트 영역을 자동으로 추출하고 면적을 계측하는 시스템

## 📌 프로젝트 개요

### 목표
AOI(자동 광학 검사) 장비가 촬영한 **False Color (Height Map)** 이미지에서:
- 솔더 페이스트 영역만 정확히 분리
- 면적을 mm² 단위로 계측
- 1000종 이상의 부품에 대응 가능한 범용 알고리즘 개발

### 핵심 기술
- **OpenCV**: 이미지 처리 및 세그멘테이션
- **HSV 색공간 분석**: 높이 정보를 색상으로 인코딩된 이미지 처리
- **K-means 클러스터링**: 자동 영역 분류
- **모폴로지 연산**: 노이즈 제거 및 정밀도 향상

---

## 🎨 Height Map 이미지란?

AOI 장비는 3D 높이 정보를 **색상으로 인코딩**합니다:
- 🔴 **빨강/오렌지**: 높은 부분 (솔더 페이스트가 솟아있음)
- 🔵 **파랑/검정**: 낮은 부분 (기판 표면)
- 🟢 **녹색**: 중간 높이

→ 일반 RGB 이미지가 아니므로 특수한 처리 필요!

---

## 🔬 3가지 세그멘테이션 방법

### 방법 1: HSV 색상 범위 추출 ⭐ (추천)
```python
# 빨강-오렌지 계열 = 높은 부분 = 솔더
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_red_orange, upper_red_orange)
```

**장점**: 간단하고 빠름, 파라미터 조정 용이
**단점**: 조명 변화에 민감할 수 있음

### 방법 2: R-B 채널 차이
```python
# R(빨강) - B(파랑) 차이가 크면 솔더
r, g, b = cv2.split(image)
diff = r - b
mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
```

**장점**: 색상 조합으로 더 정확
**단점**: 임계값 조정 필요

### 방법 3: K-means 클러스터링
```python
# 자동으로 "높은 부분"과 "낮은 부분" 분류
labels = cv2.kmeans(pixels, k=2, ...)
mask = (labels == brightest_cluster)
```

**장점**: 파라미터 최소화, 자동 분류
**단점**: 계산 비용 높음

---

## 🚀 사용법

### 1. 전체 이미지 분석 (기본)

```bash
# 프로젝트 루트에서 실행
python projects/01-aoi-solder-paste/src/analyze_images.py
```

**입력**: `data/images/*.png`
**출력**: `projects/01-aoi-solder-paste/outputs/result_*.png`

결과 이미지에는 다음이 포함됩니다:
- 원본 이미지
- HSV 색공간 변환
- R-B 채널 차이
- 3가지 방법의 세그멘테이션 결과 + 면적 계측

### 2. 단일 이미지 테스트

```python
from src.analyze_images import SolderSegmentation

# 이미지 로드
seg = SolderSegmentation("data/images/image_00002.png")

# 방법 1 실행
mask = seg.method1_hsv_color()

# 면적 계산
area = seg.calculate_area(mask)
print(f"솔더 면적: {area['total_mm2']} mm²")

# 결과 시각화
seg.visualize_all(save_path="outputs/test.png")
```

### 3. 파라미터 조정

```python
# HSV 범위 조정
mask = seg.method1_hsv_color(
    lower_hue=0,      # 빨강 시작
    upper_hue=30,     # 오렌지 끝
    lower_sat=100,    # 채도 최소
    upper_sat=255,    # 채도 최대
    lower_val=100,    # 명도 최소
    upper_val=255     # 명도 최대
)

# R-B 차이 임계값 조정
mask = seg.method2_channel_diff(threshold=30)

# K-means 클러스터 개수 조정
mask = seg.method3_kmeans(k=2)
```

---

## 📊 결과 예시

각 이미지마다 다음 정보가 출력됩니다:

```
🔍 분석 중: image_00002.png

Method 1 (HSV): 1234 px | 0.0264 mm²
Method 2 (R-B): 1189 px | 0.0255 mm²
Method 3 (K-means): 1256 px | 0.0269 mm²

✅ 결과 저장: outputs/result_image_00002.png
```

---

## 🎓 학습 포인트 (면접 대비)

### Q1: 왜 HSV 색공간을 사용했나요?
```
RGB는 조명 변화 시 R, G, B 모두 영향을 받습니다.
HSV는 밝기(V)와 색상(H)이 분리되어 있어,
조명이 바뀌어도 색상 기반 세그멘테이션이 안정적입니다.

특히 이 프로젝트의 Height Map은 높이를 색상으로 표현하므로,
H(Hue) 채널로 높이 정보를 직접 추출할 수 있습니다.
```

### Q2: 모폴로지 연산을 왜 사용하나요?
```
세그멘테이션 후 작은 노이즈 점들이 남을 수 있습니다.
- Opening (침식→팽창): 작은 흰색 점 제거
- Closing (팽창→침식): 솔더 내부의 구멍 메우기

결과적으로 더 깔끔하고 정확한 마스크를 얻을 수 있습니다.
```

### Q3: 1000종 부품을 어떻게 대응하나요?
```
이 프로젝트는 Phase 1(MVP)로, 색상 기반 세그멘테이션을 검증합니다.
향후 Phase 2에서:
- 부품 카테고리별 파라미터 프리셋 (10-15개)
- CSV 데이터의 부품종 정보로 자동 매칭
- 랜드 크기 대비 비율 기반 판정

으로 확장할 예정입니다.
```

---

## 📁 프로젝트 구조

```
01-aoi-solder-paste/
├── src/
│   ├── __init__.py
│   └── analyze_images.py      # 메인 분석 스크립트
├── outputs/                    # 결과 이미지 저장
├── notebooks/                  # Jupyter 실험 노트북
├── tests/                      # 유닛 테스트
├── docs/                       # 상세 문서
└── README.md                   # 이 파일
```

---

## 🔧 환경 요구사항

- Python 3.12+
- OpenCV 4.10+
- NumPy 1.26+
- Matplotlib 3.8+

설치:
```bash
uv sync
```

---

## 📝 다음 단계 (Phase 2)

- [ ] CSV 데이터 연동
- [ ] 부품 카테고리별 파라미터 프리셋
- [ ] 크롭 좌표 이상치 감지
- [ ] 조명 보정 (CLAHE, 적응적 이진화)
- [ ] 배치 처리 및 통계 분석
- [ ] FastAPI로 REST API 서빙
- [ ] ONNX 변환 및 엣지 배포

---

**Built with Agent Teams** - Computer Vision Portfolio Hub
