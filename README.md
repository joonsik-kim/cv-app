# CV-App: Computer Vision Portfolio Hub

> Multi-project computer vision portfolio with Agent Teams collaborative development

## 🎯 Project Structure

```
cv-app/
├── projects/              # Individual CV portfolio projects
│   └── 01-image-filters/  # First project: OpenCV filters
├── shared/                # Shared utilities across projects
│   └── cv_utils/         # Common CV functions
├── agent-system/          # Agent framework (perception-reasoning-action)
├── docs/                  # Portfolio documentation
└── .github/workflows/     # CI/CD automation
```

## 🏗️ Architecture

### Hybrid Monorepo
- **projects/** - Multiple independent CV projects (Phase 1: OpenCV → Phase 3: Deep Learning)
- **shared/** - Reusable utilities and common functions
- **agent-system/** - Optional agent framework for demonstrating AI architecture

### Agent Teams Development
- 🏗️ **Architect** - System design
- 👁️ **CV Specialist** - OpenCV implementation
- 🧠 **ML Engineer** - Deep learning (Phase 2+)
- 🚀 **DevOps** - Git, testing, deployment
- 📝 **Documentation** - Interview-ready docs

## 🚀 Getting Started

```bash
# Install dependencies
uv sync

# Run first project
cd projects/01-image-filters
python src/main.py
```

## 📚 Projects

### Phase 1: OpenCV Fundamentals
- [ ] **01-image-filters** - Spatial filtering and enhancement
- [ ] **02-feature-detection** - SIFT, ORB, keypoint matching
- [ ] **03-face-detection** - Haar cascades

### Phase 2: Hybrid (OpenCV + DL)
- [ ] **04-pretrained-models** - YOLO integration
- [ ] **05-video-analysis** - Real-time processing

### Phase 3: Deep Learning
- [ ] **06-custom-training** - Fine-tune models
- [ ] **07-segmentation** - Semantic segmentation

## 🛠️ Tech Stack

- **Python 3.12** with uv package manager
- **OpenCV** for computer vision
- **PyTorch** (Phase 2+) for deep learning
- **Jupyter** for experimentation
- **pytest** for testing
- **GitHub Actions** for CI/CD

## 📖 Documentation

- [Architecture Guide](docs/architecture.md) - System design decisions
- [Learning Path](docs/learning-path.md) - My learning journey
- [Interview Guide](docs/interview-guide.md) - Interview preparation

## 🎓 Goal

Build a professional computer vision portfolio for job interviews by March 2026, demonstrating:
- Multi-project management
- Agent Teams collaborative development
- Evolution from OpenCV to deep learning
- Professional Git workflow

---

**Built with Agent Teams** - Collaborative AI development with Claude Code
