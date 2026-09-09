#ifndef INSPIRECV_BACKENDS_OKCV_KERNELS_X86_U8C3_OPS_H_
#define INSPIRECV_BACKENDS_OKCV_KERNELS_X86_U8C3_OPS_H_

#include <cstdint>

namespace okcv {
namespace x86 {

void SwapRbU8C3(const uint8_t* source, uint8_t* destination,
                int width, int height);

void FlipHorizontalU8C3(const uint8_t* source, uint8_t* destination,
                        int width, int height);

void Rotate90U8C3(const uint8_t* source, uint8_t* destination,
                  int width, int height);

}  // namespace x86
}  // namespace okcv

#endif  // INSPIRECV_BACKENDS_OKCV_KERNELS_X86_U8C3_OPS_H_
