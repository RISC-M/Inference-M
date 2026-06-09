import tvm
from tvm import te
import os

# ===========================================================================
# 1. KERNEL GENERATORS 
# ===========================================================================

def create_matmul(N=32):
    """Standard Matrix Multiplication (Linear Layers, QKV Projections)"""
    A = te.placeholder((N, N), name='A', dtype='float32')
    B = te.placeholder((N, N), name='B', dtype='float32')
    k = te.reduce_axis((0, N), name='k')
    
    C = te.compute(
        (N, N), 
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), 
        name='C'
    )
    
    prim_func = te.create_prim_func([A, B, C]).with_attr("global_symbol", "tvm_matmul")
    mod = tvm.IRModule({"tvm_matmul": prim_func})
    
    sch = tvm.s_tir.Schedule(mod)
    sch.work_on("tvm_matmul")
    
    block_c = sch.get_sblock("C")
    
    # RESTORED: Your exact loop extraction logic
    try:
        i, j, k_axis = sch.get_loops(block_c)
    except AttributeError:
        i, j, k_axis = sch.get_sloops(block_c)
        
    vector_split_factor = 4
    j_outer, j_inner = sch.split(j, factors=[None, vector_split_factor])
    
    sch.reorder(i, j_outer, k_axis, j_inner)
    sch.vectorize(j_inner)
    
    return sch.mod, "tvm_matmul"


def create_residual_add(N=1024):
    """Vector Addition (Transformer Residual Connections)"""
    A = te.placeholder((N,), name='A', dtype='float32')
    B = te.placeholder((N,), name='B', dtype='float32')
    C = te.compute((N,), lambda i: A[i] + B[i], name='C')
    
    prim_func = te.create_prim_func([A, B, C]).with_attr("global_symbol", "tvm_residual_add")
    mod = tvm.IRModule({"tvm_residual_add": prim_func})
    
    sch = tvm.s_tir.Schedule(mod)
    sch.work_on("tvm_residual_add")
    
    block_c = sch.get_sblock("C")
    
    # RESTORED: Your exact loop extraction logic
    try:
        i, = sch.get_loops(block_c)
    except AttributeError:
        i, = sch.get_sloops(block_c)
        
    i_outer, i_inner = sch.split(i, factors=[None, 8])
    sch.vectorize(i_inner)
    
    return sch.mod, "tvm_residual_add"


def create_relu(N=1024):
    """ReLU Activation (Often used in MLP layers)"""
    A = te.placeholder((N,), name='A', dtype='float32')
    
    # Safe fallback using te.if_then_else
    C = te.compute((N,), lambda i: tvm.te.if_then_else(A[i] > 0.0, A[i], 0.0), name='C')
    
    prim_func = te.create_prim_func([A, C]).with_attr("global_symbol", "tvm_relu")
    mod = tvm.IRModule({"tvm_relu": prim_func})
    
    sch = tvm.s_tir.Schedule(mod)
    sch.work_on("tvm_relu")
    
    block_c = sch.get_sblock("C")
    
    # RESTORED: Your exact loop extraction logic
    try:
        i, = sch.get_loops(block_c)
    except AttributeError:
        i, = sch.get_sloops(block_c)
        
    i_outer, i_inner = sch.split(i, factors=[None, 8])
    sch.vectorize(i_inner)
    
    return sch.mod, "tvm_relu"


# ===========================================================================
# 2. COMPILATION ENGINE
# ===========================================================================

def compile_to_riscv(mod, func_name, output_dir="."):
    """Takes a TVM IRModule and compiles it to RISC-V assembly."""
    
    target_config = {
        "kind": "llvm",
        "mtriple": "riscv32-unknown-none-elf",
        "mcpu": "generic-rv32",
        "mattr": ["+m", "+f", "+zve32f"], 
        "mabi": "ilp32f"
    }
    target = tvm.target.Target(target_config)
    
    print(f"Compiling [{func_name}]...")
    lib = tvm.build(mod, target=target)
    
    output_file = os.path.join(output_dir, f"{func_name}.s")
    lib.write_to_file(output_file, "asm")
    print(f"  -> Saved to {output_file}")


# ===========================================================================
# 3. EXECUTION BLOCK
# ===========================================================================

if __name__ == "__main__":
    print("--- Inference-M TVM Compiler ---")
    
    kernels_to_build = [
        create_matmul(),
        create_residual_add(),
        create_relu()
    ]
    
    for mod, name in kernels_to_build:
        compile_to_riscv(mod, name)
        
    print("\nAll kernels compiled successfully!")
