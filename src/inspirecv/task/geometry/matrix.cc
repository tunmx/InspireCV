#include <inspirecv/task/core/matrix.h>
#include <inspirecv/task/geometry/projective_solver.h>

namespace inspirecv {
namespace task {

bool Matrix::setPolyToPoly(const Point source[], const Point destination[], int count) {
    if (count < 0 || count > 4) {
        INSPIRECV_TASK_ERROR("---::setPolyToPoly count out of range %d\n", count);
        return false;
    }
    if (count == 0) {
        reset();
        return true;
    }
    if (count == 1) {
        setTranslate(destination->fX - source->fX, destination->fY - source->fY);
        return true;
    }

    float coefficients[9];
    if (!geometry_internal::BuildPointBasis(source, count, coefficients)) {
        return false;
    }
    Matrix sourceBasis;
    sourceBasis.set9(coefficients);
    Matrix inverseSource;
    if (!sourceBasis.invert(&inverseSource)) {
        return false;
    }

    if (!geometry_internal::BuildPointBasis(destination, count, coefficients)) {
        return false;
    }
    Matrix destinationBasis;
    destinationBasis.set9(coefficients);
    setConcat(destinationBasis, inverseSource);
    return true;
}

}  // namespace task
}  // namespace inspirecv
