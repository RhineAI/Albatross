#!/usr/bin/env bash
set -e


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEMS_DIR="$ROOT_DIR/gems"

mkdir -p "$GEMS_DIR"

# =========================
# Clone & build libtriton_jit
# =========================
cd "$GEMS_DIR"
LIBTRITONJIT_DIR="$GEMS_DIR/libtriton_jit"
if [ ! -d "$LIBTRITONJIT_DIR" ]; then
    echo "Cloning libtriton_jit..."
    git clone https://github.com/wlxjhyf/libtriton_jit.git
else
    echo "libtriton_jit already exists, skipping clone"
fi

echo "Building libtriton_jit..."
cd "$LIBTRITONJIT_DIR"
cmake -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
      -DCUDA_INCLUDE_DIRS=$CONDA_PREFIX/include \
      -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
      -DTorch_DIR=$CONDA_PREFIX/lib/python3.13/site-packages/torch/share/cmake/Torch \
      -DCMAKE_FIND_ROOT_PATH=$CONDA_PREFIX \
      -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
      -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
      -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
      -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
      -DCUDA_CUDA_LIBRARY=$CONDA_PREFIX/lib/libcuda.so \
      -DCUDA_CUDART_LIBRARY=$CONDA_PREFIX/lib/libcudart.so \
      -S . -B build/ -DPython_ROOT="$(which python)/../.." -DCMAKE_INSTALL_PREFIX=../local
cmake -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
      -DCUDA_INCLUDE_DIRS=$CONDA_PREFIX/include \
      -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
      -DTorch_DIR=$CONDA_PREFIX/lib/python3.13/site-packages/torch/share/cmake/Torch \
      -DCMAKE_FIND_ROOT_PATH=$CONDA_PREFIX \
      -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
      -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
      -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
      -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
      -DCUDA_CUDA_LIBRARY=$CONDA_PREFIX/lib/libcuda.so \
      -DCUDA_CUDART_LIBRARY=$CONDA_PREFIX/lib/libcudart.so \
      --build build/ --parallel
cmake -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
      -DCUDA_INCLUDE_DIRS=$CONDA_PREFIX/include \
      -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
      -DTorch_DIR=$CONDA_PREFIX/lib/python3.13/site-packages/torch/share/cmake/Torch \
      -DCMAKE_FIND_ROOT_PATH=$CONDA_PREFIX \
      -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
      -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
      -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
      -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
      -DCUDA_CUDA_LIBRARY=$CONDA_PREFIX/lib/libcuda.so \
      -DCUDA_CUDART_LIBRARY=$CONDA_PREFIX/lib/libcudart.so \
      --install build/

# =========================
# Clone & install FlagGems
# =========================
cd "$GEMS_DIR"
FLAGGEMS_DIR="$GEMS_DIR/FlagGems"
if [ ! -d "$FLAGGEMS_DIR" ]; then
    echo "Cloning FlagGems..."
    git clone https://github.com/wlxjhyf/FlagGems.git
else
    echo "FlagGems already exists, skipping clone"
fi

echo "Installing FlagGems..."
cd "$FLAGGEMS_DIR"
git switch rwkv
LIBTRITONJIT_DIR_ABS="$GEMS_DIR/local"
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON \
-DDFLAGGEMS_USE_EXTERNAL_TRITON_JIT=ON \
-DTritonJIT_ROOT=$LIBTRITONJIT_DIR_ABS" \
pip install --no-build-isolation -v -e .


echo "Setup Complete!"
