# Day 1 학습 리포트: AOI 솔더 필렛 계측

> 오늘 코드에서 사용된 개념들을 정리한 자습 가이드입니다.
> 각 개념마다 "왜 쓰는지 → 어떻게 동작하는지 → 코드에서 어디에 쓰였는지" 순서로 정리했습니다.

---

## 1. 색공간 (Color Space)

### 왜 알아야 하나?
우리 코드의 핵심은 "색상으로 영역을 구분"하는 것입니다. 같은 이미지도 색공간에 따라 다르게 표현되고, 각 색공간마다 잘하는 일이 다릅니다.

### BGR (OpenCV 기본)
- OpenCV가 이미지를 읽으면 **Blue, Green, Red** 순서로 저장
- 일반적인 RGB와 순서만 반대 (주의!)
- `cv2.imread()`로 읽은 이미지는 항상 BGR

```python
img = cv2.imread("image.png")  # BGR로 읽힘
# img.shape = (높이, 너비, 3)  ← 3채널 = B, G, R
```

### HSV (Hue, Saturation, Value)
- **H (색상)**: 0~180 (OpenCV 기준). 빨강=0, 녹색=60, 파랑=120
- **S (채도)**: 0~255. 0=회색, 255=선명한 색
- **V (밝기)**: 0~255. 0=검정, 255=밝음

**왜 쓰나?**: "녹색 기판을 찾아라" 같은 작업에 최적. H 값만 보면 색상 종류를 바로 알 수 있음.

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 우리 코드에서:
# - 기판 찾기: H=35~95 (녹색~청록)
# - 필렛 엣지 찾기: H=85~130 (파란색 = 급경사)
# - 바디 제거: V < 40 (어두운 영역)
```

**코드 위치**: `analyze_images.py` 104~111행

### Lab (CIELAB)
- **L (밝기)**: 0~100
- **a (녹-빨 축)**: -128~127
- **b (파-노 축)**: -128~127

**왜 쓰나?**: "두 색이 얼마나 다른가?"를 수치로 계산할 때 가장 정확. 사람 눈의 색 인식과 거의 일치.

```python
lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

# 우리 코드에서:
# - 기판 기준색과 각 픽셀의 "색 차이(거리)"를 계산
# - 거리가 크면 = 기판이 아닌 영역 (솔더/부품)
```

**코드 위치**: `analyze_images.py` 50행, 96행

### 자습 과제
- [ ] OpenCV 공식 문서에서 `cvtColor` 함수 읽기
- [ ] HSV 색상환(color wheel) 이미지 검색해서 H 값과 색상 대응 확인
- [ ] Lab 색공간이 RGB/HSV보다 "거리 계산"에 적합한 이유 찾아보기

### 참고 자료
- OpenCV 색공간 변환: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
- HSV 색공간 시각적 설명: "opencv hsv color space tutorial" 검색
- Lab 색공간 설명: "CIELAB color space explained" 검색

---

## 2. 이미지 이진화 (Thresholding)

### 왜 알아야 하나?
"이 픽셀은 솔더인가 아닌가?" → 0 또는 255로 판별해야 함. 이것이 이진화.

### cv2.threshold()
```python
# 기본 형태
_, binary = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

# threshold_value보다 크면 → 255 (흰색)
# threshold_value보다 작으면 → 0 (검정)
```

### Otsu 이진화 (자동 임계값)
- 사람이 threshold 값을 정하는 대신 **알고리즘이 자동으로 최적값을 찾아줌**
- 원리: 히스토그램에서 두 그룹(전경/배경)의 분산이 최소가 되는 값을 선택

```python
_, binary = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                                    ↑
#                              0을 넣어도 Otsu가 자동으로 최적값 계산
```

**코드 위치**: `analyze_images.py` 101행

### cv2.inRange() (범위 기반 이진화)
- 특정 범위 안에 있는 픽셀만 255, 나머지 0

```python
# "HSV에서 파란색 범위에 해당하는 픽셀만 찾아라"
blue_mask = cv2.inRange(hsv, np.array([85, 20, 40]), np.array([130, 255, 255]))
#                            최소 [H, S, V]         최대 [H, S, V]
```

**코드 위치**: `analyze_images.py` 39~43행 (기판), 111행 (파란색)

### 자습 과제
- [ ] Otsu 알고리즘의 원리 검색 (히스토그램 이봉 분포)
- [ ] `cv2.inRange()`로 특정 색상 추출하는 예제 따라해보기
- [ ] threshold 값을 수동으로 바꿔보며 결과 차이 확인

### 참고 자료
- OpenCV 이진화: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- "opencv otsu threshold python tutorial" 검색

---

## 3. 모폴로지 연산 (Morphological Operations)

### 왜 알아야 하나?
이진화 후 노이즈(작은 점)가 남거나, 구멍이 생김. 이걸 정리하는 도구.

### 침식 (Erosion) & 팽창 (Dilation)
```
원본:       침식:       팽창:
■■■■■      ·■■■·      ■■■■■■■
■■■■■  →   ·■■■·  →   ■■■■■■■
■■■■■      ·■■■·      ■■■■■■■
                       (외곽이 1px씩 줄거나 늘어남)
```

### Open (침식 → 팽창)
- 작은 노이즈 점 제거 (먼저 침식으로 점을 없앤 후 팽창으로 원래 크기 복원)

### Close (팽창 → 침식)
- 작은 구멍 메우기 (먼저 팽창으로 구멍을 메운 후 침식으로 원래 크기 복원)

```python
kernel = np.ones((3, 3), np.uint8)  # 3x3 정사각형 커널

# 노이즈 제거 (Open → Close 순서)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 점 노이즈 제거
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 구멍 메우기
```

### 팽창 (Dilation) 단독 사용
우리 코드에서 **파란색 영역을 확장**할 때 사용:
```python
dilate_kernel = np.ones((7, 7), np.uint8)
blue_expanded = cv2.dilate(blue_mask, dilate_kernel, iterations=2)
# 파란색 영역이 7x7 커널로 2번 팽창 → 주변 14px까지 확장
```

**코드 위치**: `analyze_images.py` 114~115행, 122~124행

### 자습 과제
- [ ] 커널 크기(3x3 vs 5x5 vs 7x7)에 따른 결과 차이 실험
- [ ] Open과 Close의 순서를 바꾸면 어떻게 되는지 확인
- [ ] `iterations` 값을 바꿔보기 (1 vs 2 vs 3)

### 참고 자료
- OpenCV 모폴로지: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
- "opencv morphology open close python" 검색

---

## 4. 컨투어 (Contour Detection)

### 왜 알아야 하나?
이진 마스크에서 "덩어리"를 찾고, 그 면적을 계산하기 위해.

### 기본 사용법
```python
# 이진 이미지에서 외곽선 찾기
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                                           ↑ 외곽 컨투어만     ↑ 꼭짓점만 저장 (메모리 절약)

# contours = [컨투어1, 컨투어2, ...] 리스트
# 각 컨투어 = 점들의 좌표 배열

# 면적 계산
area = cv2.contourArea(contour)

# 마스크에 컨투어 채워 그리기
cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
```

**코드 위치**: `analyze_images.py` 127~137행

### 자습 과제
- [ ] `cv2.RETR_EXTERNAL` vs `cv2.RETR_TREE`의 차이
- [ ] `cv2.contourArea()` 외에 `cv2.arcLength()` (둘레) 도 사용해보기
- [ ] 가장 큰 컨투어만 남기는 코드 작성해보기

### 참고 자료
- OpenCV 컨투어: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

---

## 5. Color Distance (색 거리 계산)

### 왜 알아야 하나?
"이 픽셀이 기판색과 얼마나 다른가?"를 숫자로 계산하는 핵심 기법.

### 유클리드 거리 (Euclidean Distance)
```
두 점 사이의 직선 거리:
  거리 = sqrt((x1-x2)² + (y1-y2)² + (z1-z2)²)

Lab 색공간에서:
  색 차이 = sqrt((L1-L2)² + (a1-a2)² + (b1-b2)²)
```

### 우리 코드에서의 사용
```python
# 기판 기준색 (OG에서 추출)
board_ref_lab = [21, 118, 133]  # L, a, b

# 모든 픽셀과의 거리 계산 (벡터 연산 = 한번에 모든 픽셀)
diff = lab_image - board_ref_lab          # 각 채널별 차이
distance = np.sqrt(np.sum(diff**2, axis=2))  # 유클리드 거리

# 거리가 크면 = 기판과 다른 색 = 솔더 영역
```

**코드 위치**: `analyze_images.py` 96~98행

### 왜 Lab에서 하나? (RGB/HSV가 아니라)
- RGB: 밝기와 색상이 섞여있어서 거리 계산이 부정확
- HSV: H가 각도(0°=360°=빨강)라서 유클리드 거리 부적합
- Lab: 사람 눈의 색 인식과 거리 계산이 선형적으로 일치 → 가장 정확

### 자습 과제
- [ ] 유클리드 거리를 직접 손으로 계산해보기 (2D 예제)
- [ ] Lab 색공간에서 ΔE (Delta E) 개념 검색
- [ ] numpy 브로드캐스팅이 뭔지 찾아보기 (lab_image - board_ref_lab 이 왜 동작하는지)

### 참고 자료
- "color difference delta e lab" 검색
- NumPy 브로드캐스팅: https://numpy.org/doc/stable/user/basics.broadcasting.html

---

## 6. Git 브랜치 관리

### 왜 알아야 하나?
코드를 실험적으로 바꿀 때 기존 코드를 안전하게 보존하기 위해.

### 오늘 사용한 명령어

```bash
# 1. 새 브랜치 생성 + 이동
git checkout -b feature/hybrid-approach
# master에서 갈라져 나온 새 브랜치에서 작업 시작

# 2. 변경 파일 스테이징 + 커밋
git add .gitignore analyze_images.py
git commit -m "feat: hybrid 접근법 구현"

# 3. GitHub에 올리기
git push origin master                    # master 브랜치 push
git push -u origin feature/hybrid-approach  # feature 브랜치 push (Publish Branch)

# 4. 상태 확인
git status          # 변경된 파일 확인
git branch -vv      # 모든 브랜치 + 리모트 추적 상태
git log --oneline   # 커밋 이력
```

### .gitignore
```
# 이 파일에 적힌 패턴과 일치하는 파일은 Git이 무시함
data/imgae_processing/   # 이미지 데이터 (용량 큼, Git에 안 올림)
outputs/                  # 결과물 (매번 새로 생성)
```

### 브랜치 전략 요약
```
master: 완성된 코드만 (직접 커밋 X, merge만)
feature/*: 실험/개발용 (자유롭게 커밋)
  - 성공 → master에 merge
  - 실패 → 브랜치만 삭제
```

### 자습 과제
- [ ] `git merge`와 `git rebase`의 차이 검색
- [ ] `git log --graph --all --oneline`으로 브랜치 시각화 해보기
- [ ] VS Code Git Graph 확장 프로그램 설치해서 시각적으로 확인

### 참고 자료
- Git 브랜치 기초: https://git-scm.com/book/ko/v2/Git-브랜치-브랜치란-무엇인가
- "git branch workflow for beginners" 검색

---

## 7. 오늘 코드 전체 흐름 (파이프라인)

```
[입력]
  OG 이미지 (부품 전체 + 기판 배경)
  Cropped 이미지 (ROI = 한 핀의 필렛 영역)

      ↓

[Step 1] OG에서 기판색 추출
  OG → HSV 변환 → H=35~95 (녹색) 마스크 → Lab 변환 → 기판 픽셀 중앙값
  결과: board_ref_lab = [21, 118, 133]

      ↓

[Step 2] Cropped에서 기판 제거
  Cropped → Lab 변환 → 기판색과의 유클리드 거리 계산 → Otsu 이진화
  결과: non_board 마스크 (기판=0, 나머지=255)

      ↓

[Step 3] 바디 제거
  Cropped → HSV 변환 → V < 40인 픽셀 제거 (어두운 부품 바디)
  결과: not_dark 마스크

      ↓

[Step 4] 필렛 분리
  HSV → H=85~130 (파란색 = 급경사) 마스크 → 7x7 커널로 2회 팽창
  최종 = non_board AND not_dark AND blue_expanded
  결과: fillet_mask

      ↓

[Step 5] 면적 계산
  컨투어 검출 → 면적 합산 → mm² 변환 (1px = 0.01465mm)

      ↓

[출력]
  각 ROI별 필렛 면적 (px, mm²) + 시각화 이미지
```

---

## 내일 할 일

1. **위 개념들 자습** (특히 색공간, 이진화, 모폴로지)
2. **코드를 직접 읽으면서** 각 단계의 입출력 확인
3. **파라미터 실험**: 커널 크기, threshold 값 등을 바꿔보며 결과 변화 관찰
4. **다른 부품 타입 테스트** 준비 (812100-29910730: 1핀 62장)

---

## 면접 대비 포인트

> Q: 왜 RGB가 아닌 Lab 색공간을 사용했나요?
> A: RGB에서의 유클리드 거리는 사람의 색 인식과 불일치합니다. Lab은 perceptually uniform하여 색 차이(ΔE)가 인간 시각과 선형적으로 대응합니다.

> Q: Otsu 이진화를 선택한 이유는?
> A: 부품마다 색상 분포가 달라서 고정 threshold가 불가능합니다. Otsu는 히스토그램 기반으로 자동 최적 임계값을 계산하므로 부품 간 일반화가 가능합니다.

> Q: 왜 파란색으로 필렛을 찾나요?
> A: OMRON AOI의 false color는 경사도 기반입니다. 파란색=급경사는 솔더 필렛의 곡면에서만 나타나는 고유 특성이며, 랜드(평탄)나 기판에는 없습니다.

> Q: OG와 Cropped를 분리한 이유는?
> A: Cropped(ROI)는 크기가 작고(~30x55px) 기판이 충분히 보이지 않아 기준색 추출이 불안정합니다. OG는 기판 배경이 넓게 보여 안정적인 기준색 추출이 가능합니다.
