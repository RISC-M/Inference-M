#!/bin/bash
set -e
# 1. Point to the source
export TVM_HOME=$(realpath ../tvm_src)

# 2. ONLY add the main tvm python path. 
# Do NOT add the tvm-ffi path here. Let the virtual environment handle it!
export PYTHONPATH=$TVM_HOME/python

# 3. Point to the compiled C++ libraries
export TVM_LIBRARY_PATH=$TVM_HOME/build/lib
export LD_LIBRARY_PATH=$TVM_HOME/build/lib:${LD_LIBRARY_PATH}

echo "Environment set to: $TVM_HOME"
echo "1. Generating RISC-V Assembly via TVM..."
python3 compile.py

echo "2. Preparing and compiling baremetal wrappers..."

# THE FIX: Rename the internal TVM symbol to match what main.c expects
sed -i 's/__tvm_ffi_tvm_matmul/tvm_matmul/g' tvm_matmul.s
# Also force it to be a global symbol so the linker can find it
sed -i 's/^tvm_matmul:/ .globl tvm_matmul\ntvm_matmul:/' tvm_matmul.s

# Assemble the patched assembly file
clang-17 --target=riscv32-unknown-none-elf -march=rv32im_zve32x -mabi=ilp32 -c tvm_matmul.s -o tvm_matmul.o

# Paths for TVM/FFI headers
INCLUDES="-I./tvm_src/include \
          -I./tvm_src/3rdparty/tvm-ffi/include \
          -I./tvm_src/3rdparty/dlpack/include \
          -I./tvm_src/3rdparty/tvm-ffi/3rdparty/dlpack/include \
          -I./tvm_src/3rdparty/dmlc-core/include \
          -I./tvm_src/3rdparty/tvm-ffi/3rdparty/dmlc-core/include"

# Compile wrappers
# Add -fno-builtin to the clang command
clang-17 $INCLUDES --target=riscv32-unknown-none-elf -march=rv32im_zve32x -mabi=ilp32 -O3 -fno-builtin -c main.c -o main.o
clang-17 --target=riscv32-unknown-none-elf -march=rv32im_zve32x -mabi=ilp32 -c start.S -o start.o

# Update the linking step in build.sh to this:
echo "3. Linking baremetal ELF..."
# The -u flag forces the linker to include the symbols 'tohost' and 'fromhost'
ld.lld-17 -T link.ld -u tohost -u fromhost start.o main.o tvm_matmul.o -o kernel.elf

echo "4. Simulating accelerator execution..."
spike --isa=rv32im_zve32x kernel.elf

echo "Pipeline complete! The kernel.elf is fully linked."
