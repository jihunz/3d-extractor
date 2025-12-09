# 3D Extractor

SAM3 + SAM 3D Objects 기반 이미지에서 3D Gaussian Splatting 추출 서버

## 🚀 Quick Start

```bash
# 의존성 설치
make install
# 또는
pip install -r requirements.txt

# 개발 서버 실행
make dev
# 또는
python run.py

# 브라우저에서 접속
open http://localhost:8000
```

## 📋 Features

- **이미지 업로드** → **클릭으로 마스크 생성** → **3D Gaussian Splat 추출**
- SAM3 기반 인터랙티브 세그멘테이션 (포인트/박스 프롬프트)
- SAM 3D Objects 기반 3D 재구성
- GaussianSplats3D 기반 인터랙티브 3D 뷰어
- PLY 파일 다운로드

## 🛠️ Commands

```bash
make help      # 사용 가능한 명령어 보기
make install   # 의존성 설치
make dev       # 개발 서버 (auto-reload)
make prod      # 프로덕션 서버
make clean     # 캐시 및 임시 파일 정리
make test      # 테스트 실행
```

또는 Python 스크립트 직접 실행:

```bash
python run.py              # 개발 모드
python run.py --prod       # 프로덕션 모드
python run.py --port 8080  # 커스텀 포트
python run.py --help       # 도움말
```

## 📁 Project Structure

```
3d-extractor/
├── main.py              # FastAPI 메인
├── run.py               # 실행 스크립트
├── Makefile             # Make 명령어
├── requirements.txt     # 의존성
├── models/
│   ├── sam3_model.py    # SAM3 래퍼
│   └── sam3d_model.py   # SAM 3D Objects 래퍼
├── routers/
│   ├── segment.py       # 세그멘테이션 API
│   └── reconstruct.py   # 3D 재구성 API
└── static/
    ├── index.html       # 메인 UI
    ├── app.js           # 프론트엔드 로직
    └── viewer.html      # 3D 뷰어
```

## 🔧 Full Installation (SAM3 + SAM 3D Objects)

현재는 **Mock 모드**로 실행됩니다. 실제 모델을 사용하려면:

```bash
# Conda 환경 생성
conda create -n 3d-extractor python=3.12
conda activate 3d-extractor

# PyTorch 설치 (CUDA 12.6)
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# SAM3 설치
pip install git+https://github.com/facebookresearch/sam3.git

# SAM 3D Objects 설치
pip install git+https://github.com/facebookresearch/sam-3d-objects.git

# 서버 실행
python run.py
```

## 📖 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | 메인 페이지 |
| `/viewer` | GET | 3D 뷰어 |
| `/api/segment/upload` | POST | 이미지 업로드 |
| `/api/segment/predict` | POST | 마스크 예측 |
| `/api/reconstruct/generate` | POST | 3D 생성 |
| `/api/reconstruct/download/{id}` | GET | PLY 다운로드 |
| `/api/info` | GET | 서버 정보 |
| `/health` | GET | 헬스 체크 |

## 📄 License

MIT

