#ifndef INSPIRECV_TASK_GEOMETRY_PROJECTIVE_SOLVER_H_
#define INSPIRECV_TASK_GEOMETRY_PROJECTIVE_SOLVER_H_

#include <inspirecv/task/core/rect.h>

namespace inspirecv {
namespace task {
namespace geometry_internal {

// Builds the compatibility basis used by Matrix::setPolyToPoly for two, three,
// or four control points. The output uses Matrix's public row-major 3x3 order.
bool BuildPointBasis(const Point* points, int count, float output[9]);

}  // namespace geometry_internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_GEOMETRY_PROJECTIVE_SOLVER_H_
