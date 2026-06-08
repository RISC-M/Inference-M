import tvm
from tvm import te

# ---------------------------------------------------------------------------
# 1. Define the Computation (32x32 Matrix Multiplication)
# ---------------------------------------------------------------------------
N = 32
A = te.placeholder((N, N), name='A', dtype='int32')
B = te.placeholder((N, N), name='B', dtype='int32')

k = te.reduce_axis((0, N), name='k')
C = te.compute(
    (N, N), 
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), 
    name='C'
)

# ---------------------------------------------------------------------------
# 2. Transition to TensorIR and Schedule
# ---------------------------------------------------------------------------
# The 'global_symbol' attribute here is the source of truth for the linker
prim_func = te.create_prim_func([A, B, C]).with_attr("global_symbol", "tvm_matmul")
mod = tvm.IRModule({"tvm_matmul": prim_func})

sch = tvm.s_tir.Schedule(mod)
sch.work_on("tvm_matmul")

# ---------------------------------------------------------------------------
# 3. Apply Hardware Vectorization (zve32x)
# ---------------------------------------------------------------------------
block_c = sch.get_sblock("C")

try:
    i, j, k_axis = sch.get_loops(block_c)
except AttributeError:
    i, j, k_axis = sch.get_sloops(block_c)

# Vectorize the inner loop
vector_split_factor = 4
j_outer, j_inner = sch.split(j, factors=[None, vector_split_factor])

sch.reorder(i, j_outer, k_axis, j_inner)
sch.vectorize(j_inner)

# ---------------------------------------------------------------------------
# 4. Compile directly to RISC-V LLVM Assembly
# ---------------------------------------------------------------------------
target_config = {
    "kind": "llvm",
    "mtriple": "riscv32-unknown-none-elf",
    "mcpu": "generic-rv32",
    "mattr": ["+m", "+zve32x"],
    "mabi": "ilp32"
}
target = tvm.target.Target(target_config)

print("Compiling TVM TIR schedule to RISC-V assembly...")

# Removed the 'name' argument; the global_symbol attribute handles this now
lib = tvm.build(sch.mod, target=target)

# Safely extract assembly text
output_file = "tvm_matmul.s"
print(f"Saving assembly to {output_file}...")
lib.write_to_file(output_file, "asm")

print("Compilation Successful!")
