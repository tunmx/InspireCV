#ifndef INSPIRECV_TASK_GEOMETRY_GNU_PROJECTIVE_MATH_H_
#define INSPIRECV_TASK_GEOMETRY_GNU_PROJECTIVE_MATH_H_

#include <inspirecv/core/geometry/projective2d.h>
#include <cmath>

namespace inspirecv {
namespace task {
namespace geometry_internal {

#if defined(__GNUC__) && !defined(__clang__)
void ComposeProjectiveCompat(const float left[9], const float right[9],
                             float output[9]);
#elif defined(__clang__) && defined(__aarch64__) && !INSPIRECV_TASK_PRESERVE_LTO_EVALUATION
inline void ComposeProjectiveCompat(const float left[9], const float right[9],
                                    float output[9]) {
#pragma clang fp reassociate(off) contract(off)
    // Preserve the ARM Release dot-product order independently of SLP's choice
    // of fused operand. Inlining keeps the original allocation-free hot path.
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            const float* a = left + row * 3;
            const float* b = right + column;
            const bool fused_tail = row == 0 || (row == 1 && column == 0);
            const bool reverse_first_pair =
              (row == 1 && column == 2) || (row == 2 && column != 0);
            const float first_pair = reverse_first_pair
              ? std::fma(a[0], b[0], a[1] * b[3])
              : std::fma(a[1], b[3], a[0] * b[0]);
            output[row * 3 + column] = fused_tail
              ? std::fma(a[2], b[6], first_pair)
              : first_pair + a[2] * b[6];
        }
    }
}
#else
inline void ComposeProjectiveCompat(const float left[9], const float right[9],
                                    float output[9]) {
    geometry::StoreProjective(geometry::Compose(left, right), output);
}
#endif

}  // namespace geometry_internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_GEOMETRY_GNU_PROJECTIVE_MATH_H_
