#include <stdint.h>

// --- BAREMETAL LINKER FIXES ---
void TVMFFIErrorSetRaisedFromCStrParts(const char* err, ...) {}
void TVMAPISetLastError(const char* err) {}

// Global HTIF symbols required by Spike
extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

typedef struct {
    void* data; int32_t device_type; int32_t device_id; int32_t ndim;
    int32_t dtype_code; uint8_t dtype_bits; uint16_t dtype_lanes;
    int32_t* shape; int64_t* strides; uint64_t byte_offset;
} DLTensor;

typedef union { int64_t v_int64; double v_float64; void* v_handle; const char* v_str; } TVMValue;

extern int32_t tvm_matmul(TVMValue* args, int* type_codes, int num_args, TVMValue* out_value, int* out_tcode, void* resource_handle);

int main() {
    int32_t A_data[1024];
    int32_t B_data[1024];
    int32_t C_data[1024];

    // Simple initialization loop. With -fno-builtin, the compiler 
    // is now forbidden from replacing this with a memset call.
    for (int i = 0; i < 1024; i++) {
        A_data[i] = 1;
        B_data[i] = 2;
        C_data[i] = 0;
    }

    int32_t shape[2] = {32, 32};
    DLTensor A = {A_data, 1, 0, 2, 0, 32, 1, shape, 0, 0};
    DLTensor B = {B_data, 1, 0, 2, 0, 32, 1, shape, 0, 0};
    DLTensor C = {C_data, 1, 0, 2, 0, 32, 1, shape, 0, 0};

    TVMValue args[3];
    args[0].v_handle = &A;
    args[1].v_handle = &B;
    args[2].v_handle = &C;
    int type_codes[3] = {3, 3, 3};

    tvm_matmul(args, type_codes, 3, 0, 0, 0);

    // Verify result: 1 * 2 * 32 = 64
    if (C_data[0] == 64) {
        while(1); // Success signal
    }
    return 1;
}
