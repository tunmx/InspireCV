#ifndef INSPIRECV_TASK_GEOMETRY_GNU_PROJECTIVE_MATH_H_
#define INSPIRECV_TASK_GEOMETRY_GNU_PROJECTIVE_MATH_H_

#include <inspirecv/core/geometry/projective2d.h>

namespace inspirecv {
namespace task {
namespace geometry_internal {

#if defined(__GNUC__) && !defined(__clang__)
void ComposeProjectiveCompat(const float left[9], const float right[9],
                             float output[9]);
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
