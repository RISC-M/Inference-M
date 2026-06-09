import re
import sys
from collections import Counter

def analyze_riscv_assemblies(filepaths):
    # Regex Patterns
    instruction_pattern = re.compile(r'^\s*([a-z][a-z0-9\.]+)(?:\s+|$)')
    vreg_pattern = re.compile(r'\bv([0-3]?[0-9])\b')
    vsetvli_sew_pattern = re.compile(r'\be(\d+)\b')
    vsetvli_lmul_pattern = re.compile(r'\bm(f?[1248])\b')

    # Aggregated Metrics
    vector_instruction_counts = Counter()
    scalar_instruction_counts = Counter()
    vector_registers_used = set()
    vsetvli_configs = Counter()
    
    total_instructions = 0
    total_vector_instructions = 0
    total_scalar_instructions = 0
    files_processed = 0

    for filepath in filepaths:
        try:
            with open(filepath, 'r') as f:
                files_processed += 1
                for line in f:
                    line = line.split('#')[0].split('//')[0].strip()
                    
                    if not line or line.endswith(':') or line.startswith('.'):
                        continue

                    match = instruction_pattern.match(line)
                    if match:
                        mnemonic = match.group(1)
                        total_instructions += 1

                        # Categorize as Vector or Scalar
                        if mnemonic.startswith('v'):
                            vector_instruction_counts[mnemonic] += 1
                            total_vector_instructions += 1
                        else:
                            scalar_instruction_counts[mnemonic] += 1
                            total_scalar_instructions += 1

                        # Track Vector State (vsetvli / vsetivli)
                        if mnemonic in ['vsetvli', 'vsetivli']:
                            sew = vsetvli_sew_pattern.search(line)
                            lmul = vsetvli_lmul_pattern.search(line)
                            sew_val = sew.group(1) if sew else "Unknown"
                            lmul_val = lmul.group(1) if lmul else "Unknown"
                            vsetvli_configs[f"SEW={sew_val}, LMUL={lmul_val}"] += 1

                    # Track Register Pressure
                    vregs = vreg_pattern.findall(line)
                    for reg in vregs:
                        vector_registers_used.add(int(reg))

        except FileNotFoundError:
            print(f"Warning: Could not find file '{filepath}'. Skipping.")

    # Prevent division by zero if no files were read
    if files_processed == 0:
        print("Error: No valid files were processed.")
        sys.exit(1)

    # --- Print the Report ---
    print("="*50)
    print("INFERENCE-M: RISC-V ASSEMBLY AGGREGATE PROFILE")
    print("="*50)
    
    print("\nSUMMARY:")
    print(f"Files Processed:     {files_processed}")
    print(f"Total Instructions:  {total_instructions}")
    if total_instructions > 0:
        print(f"Vector Instructions: {total_vector_instructions} ({(total_vector_instructions/total_instructions)*100:.1f}%)")
        print(f"Scalar Instructions: {total_scalar_instructions} ({(total_scalar_instructions/total_instructions)*100:.1f}%)")

    print("\nVECTOR INSTRUCTION MIX (Top 15):")
    if not vector_instruction_counts:
        print("  None detected.")
    for instr, count in vector_instruction_counts.most_common(15):
        print(f"  {instr:<12} : {count}")

    print("\nSCALAR INSTRUCTION MIX (Top 15):")
    if not scalar_instruction_counts:
        print("  None detected.")
    for instr, count in scalar_instruction_counts.most_common(15):
        print(f"  {instr:<12} : {count}")

    print("\nVECTOR STATE CONFIGURATIONS (vsetvli):")
    if not vsetvli_configs:
        print("  None detected.")
    for config, count in vsetvli_configs.most_common():
        print(f"  {config:<20} : {count} times")

    print("\nREGISTER PRESSURE:")
    print(f"  Total unique 'v' registers used: {len(vector_registers_used)} / 32")
    if vector_registers_used:
        sorted_regs = sorted(list(vector_registers_used))
        print(f"  Registers: v{', v'.join(map(str, sorted_regs))}")
    
    print("="*50)

if __name__ == "__main__":
    # Remove the script name from the arguments
    args = sys.argv[1:]
    
    if not args:
        print("Usage: python analyze_asm.py <file1.s> [file2.s] [file3.s] ...")
        print("Example: python analyze_asm.py compiler/*.s")
        sys.exit(1)
    
    analyze_riscv_assemblies(args)
