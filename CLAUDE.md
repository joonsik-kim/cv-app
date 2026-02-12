# CV-App: Computer Vision Portfolio Hub

## 프로젝트 개요
- **목표**: Agent Teams를 활용한 다중 CV 프로젝트 포트폴리오 구축
- **아키텍처**: Hybrid Monorepo (projects/ + shared/ + agent-system/)
- **마감**: 2026년 3월 (면접 준비)
- **사용자**: 비개발자, Python/Git 기초, OpenCV 기본, RTX 3050 GPU (4GB)

## 핵심 컨셉
- **여러 개의 독립적인 CV 프로젝트** (단일 프로젝트 아님!)
- **Agent Teams 협업 개발** (5개 전문 에이전트)
- **단계적 진화**: OpenCV (Phase 1) → Hybrid (Phase 2) → Deep Learning (Phase 3)
- **전문적인 Git 워크플로우** (feature 브랜치, conventional commits)

## 기술 스택
- **Core**: Python 3.12 (uv로 관리)
- **CV**: OpenCV, NumPy, Pillow, Matplotlib
- **DL** (Phase 2+): PyTorch, YOLO, HuggingFace
- **Dev**: pytest, Jupyter, black, ruff
- **CI/CD**: GitHub Actions
- **Agent Teams**: Claude Code (CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1)

## 프로젝트 구조 (Hybrid Monorepo)

```
cv-app/
├── projects/              # 🎯 메인: 독립적인 CV 프로젝트들
│   ├── 01-image-filters/
│   ├── 02-feature-detection/
│   └── 03-face-detection/
│
├── shared/                # 🔧 공유: 재사용 가능한 유틸리티
│   └── cv_utils/
│
├── agent-system/          # 🤖 에이전트 프레임워크 (선택)
│   ├── core/
│   ├── perception/
│   ├── reasoning/
│   └── action/
│
├── docs/                  # 📚 문서: 면접 준비
│   ├── architecture.md
│   ├── learning-path.md
│   └── interview-guide.md
│
├── data/                  # 📊 데이터셋
├── notebooks/             # 📓 실험 노트북
└── .github/workflows/     # ⚙️ CI/CD
```

## Agent Teams 구성

### 5개 전문 에이전트
1. **🏗️ Portfolio Architect** - 시스템 설계, 기술 선택
2. **👁️ CV Specialist** - OpenCV, 전통 CV 구현
3. **🧠 ML Engineer** - 딥러닝 통합 (Phase 2+)
4. **🚀 DevOps** - Git, 자동화, 배포
5. **📝 Documentation** - 면접용 문서화

### 협업 방식
- 병렬 작업: 각 에이전트가 독립적으로 작업
- Architect가 품질 검증
- 사용자가 최종 승인

## 완료된 작업
- [x] uv 설치 및 프로젝트 초기화
- [x] Python 3.12 가상환경 생성
- [x] OpenCV, NumPy, Pillow, Matplotlib 설치
- [x] Git 초기화 + .gitignore 설정
- [x] Agent Teams 활성화 (CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1)
- [x] MCP 서버 추가 (Playwright, Python REPL, Jupyter, mcp-vision, Docker, ML Research)
- [x] Hybrid Monorepo 구조 생성

## 진행 중인 작업
- [ ] Project 01: Image Filter Studio (1주일)
- [ ] 공유 유틸리티 작성 (shared/cv_utils/)
- [ ] 첫 Git 커밋 (professional commit message)

## 다음 할 일 (Phase 1: OpenCV 기초)
- [ ] Project 01 완성 (5-7개 필터 구현)
- [ ] Project 02: Feature Detection & Matching
- [ ] Project 03: Face Detection (Haar Cascade)
- [ ] GitHub 저장소 공개
- [ ] 면접 준비 문서 작성

## 작업 규칙
- **개념 먼저**: 코드를 바로 주지 말고 개념 먼저 설명
- **이유 포함**: 왜 이렇게 하는지 이유 포함
- **사용자 작업 명시**: 직접 해볼 부분 명시
- **면접 포인트**: 각 결정에 대한 면접 질문/답변
- **학습 자료**: 참조 학습 자료 매번 제시
- **탑다운 방식**: 먼저 작성하고 사용자가 찾아서 공부

## 환경 정보
- **OS**: Windows
- **GPU**: RTX 3050 Laptop (4GB VRAM) - Phase 2+ 딥러닝 시 사용
- **IDE**: VS Code + Claude Code CLI
- **MCP 서버**:
  - sequential-thinking, context7, memory
  - github, playwright, python-repl, jupyter
  - mcp-vision, docker, ml-research
  - brave-search, filesystem, notion

## Git 워크플로우
- **Branches**: main (production), develop (integration), project/* (features)
- **Commits**: Conventional Commits (feat, fix, docs, test, etc.)
- **태그**: v0.1.0, v0.2.0, ... (각 프로젝트 완성 시)

## 타임라인 (3월 목표)
- **Week 1**: 기반 구축 + Project 01
- **Week 2-3**: Project 02, 03
- **Week 4**: 문서화 & 포트폴리오 정리
- **3월 완성**: Phase 1 (OpenCV 기초) 3-5개 프로젝트

## Phase 4: Edge & API (선택적 확장)

### API 서버
- **FastAPI**: CV 모델을 REST API로 서빙
- **엔드포인트**: 이미지 업로드 → 처리 → 결과 반환
- **문서화**: Swagger 자동 생성
- **배포**: Docker 컨테이너

### Edge 최적화
- **ONNX 변환**: PyTorch → ONNX (범용 포맷)
- **양자화**: FP32 → INT8 (모델 크기 75% 감소)
- **프루닝**: 불필요한 파라미터 제거
- **벤치마크**: 실제 디바이스에서 성능 측정

### 타겟 디바이스
- **Raspberry Pi 4**: 저가형 엣지
- **Jetson Nano**: GPU 가속 엣지
- **AWS Lambda**: 서버리스 배포

### 추가 의존성
```bash
# API 서버
uv sync --extra api

# Edge 배포
uv sync --extra edge

# Edge GPU
uv sync --extra edge-gpu
```

## 참고
- 계획 파일: `C:\Users\kim joonsik\.claude\plans\staged-mixing-scroll.md`
- Agent Teams 설정: `C:\Users\kim joonsik\.claude\settings.json`
