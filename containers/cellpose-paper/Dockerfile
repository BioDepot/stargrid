# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel@sha256:adb0f2d3769e0796a5d86740b0f900a72835b6fa9577b77d6efed460f1322fcb

ARG OPENCV_VERSION=4.11.0
ARG CUDA_ARCH_BIN=8.9

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git libglib2.0-0 libgl1 \
        libjpeg-dev libopenjp2-7-dev libpng-dev libtiff-dev ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY containers/cellpose-paper/requirements.paper.lock /tmp/requirements.paper.lock
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.paper.lock \
    && python -m pip uninstall -y \
        opencv-contrib-python opencv-contrib-python-headless \
        opencv-python opencv-python-headless || true

RUN git clone --branch "${OPENCV_VERSION}" --depth 1 \
        https://github.com/opencv/opencv.git /tmp/opencv \
    && git clone --branch "${OPENCV_VERSION}" --depth 1 \
        https://github.com/opencv/opencv_contrib.git /tmp/opencv_contrib \
    && site_packages="$(python -c 'import site; print(site.getsitepackages()[0])')" \
    && cmake -S /tmp/opencv -B /tmp/opencv/build -G Ninja \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
        -D BUILD_LIST=core,imgproc,imgcodecs,python3,cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping \
        -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF \
        -D BUILD_opencv_java=OFF -D BUILD_opencv_python3=ON \
        -D PYTHON3_EXECUTABLE="$(command -v python)" \
        -D PYTHON3_PACKAGES_PATH="${site_packages}" \
        -D WITH_CUDA=ON -D CUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
        -D CUDA_ARCH_PTX="" -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON \
        -D WITH_CUBLAS=ON -D WITH_CUDNN=OFF -D WITH_GTK=OFF \
        -D WITH_QT=OFF -D WITH_OPENGL=OFF -D OPENCV_ENABLE_NONFREE=OFF \
    && cmake --build /tmp/opencv/build --target install --parallel "$(nproc)" \
    && ldconfig \
    && rm -rf /tmp/opencv /tmp/opencv_contrib

WORKDIR /opt/visium-hd-processing
COPY pyproject.toml README.md /opt/visium-hd-processing/
COPY src /opt/visium-hd-processing/src
COPY scripts/publication/segment_hd_he_cellpose.py \
    /opt/visium-hd-processing/scripts/publication/segment_hd_he_cellpose.py
RUN python -m pip install -e . --no-deps \
    && python - <<'PY'
import cv2
info = cv2.getBuildInformation()
assert "NVIDIA CUDA" in info
assert hasattr(cv2, "cuda_GpuMat")
assert hasattr(cv2.cuda, "createGaussianFilter")
assert hasattr(cv2.cuda, "createMorphologyFilter")
assert hasattr(cv2.cuda, "threshold")
PY

LABEL org.opencontainers.image.title="Visium HD paper Cellpose environment" \
      org.opencontainers.image.version="paper1" \
      org.opencontainers.image.source="BioDepot/visium-hd-processing"

CMD ["python", "scripts/publication/segment_hd_he_cellpose.py", "--help"]
