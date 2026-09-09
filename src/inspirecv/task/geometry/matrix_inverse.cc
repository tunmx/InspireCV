#include <inspirecv/core/geometry/affine2d.h>
#include <inspirecv/core/geometry/projective2d.h>
#include <inspirecv/task/core/matrix.h>

namespace inspirecv {
namespace task {
namespace {

geometry::Affine2D AffinePart(const Matrix& matrix) {
    return {matrix.getScaleX(), matrix.getSkewX(), matrix.getTranslateX(),
            matrix.getSkewY(), matrix.getScaleY(), matrix.getTranslateY()};
}

}  // namespace

bool Matrix::invertNonIdentity(Matrix* destination) const {
    INSPIRECV_TASK_ASSERT(!isIdentity());
    const TypeMask type = getType();

    if ((type & ~(kScale_Mask | kTranslate_Mask)) == 0) {
        if ((type & kScale_Mask) == 0) {
            if (destination != nullptr) {
                destination->setTranslate(-fMat[kMTransX], -fMat[kMTransY]);
            }
            return true;
        }

        const float scaleX = fMat[kMScaleX];
        const float scaleY = fMat[kMScaleY];
        if (scaleX == 0.0f || scaleY == 0.0f) {
            return false;
        }
        if (destination != nullptr) {
            const float inverseX = 1.0f / scaleX;
            const float inverseY = 1.0f / scaleY;
            destination->fMat[kMScaleX] = inverseX;
            destination->fMat[kMSkewX] = 0.0f;
            destination->fMat[kMTransX] = -fMat[kMTransX] * inverseX;
            destination->fMat[kMSkewY] = 0.0f;
            destination->fMat[kMScaleY] = inverseY;
            destination->fMat[kMTransY] = -fMat[kMTransY] * inverseY;
            destination->fMat[kMPersp0] = 0.0f;
            destination->fMat[kMPersp1] = 0.0f;
            destination->fMat[kMPersp2] = 1.0f;
            destination->setTypeMask(type | kRectStaysRect_Mask);
        }
        return true;
    }

    const bool perspective = (type & kPerspective_Mask) != 0;
    const double reciprocalDeterminant =
      geometry::ReciprocalDeterminant(fMat, perspective);
    if (reciprocalDeterminant == 0.0) {
        return false;
    }

    Matrix scratch;
    Matrix* output = destination == nullptr || destination == this ? &scratch : destination;
    if (perspective) {
        geometry::StoreInverseProjective(fMat, reciprocalDeterminant, output->fMat);
#if defined(__clang__) && defined(__aarch64__) && !INSPIRECV_TASK_PRESERVE_LTO_EVALUATION
        {
#pragma clang fp reassociate(off) contract(off)
            // The ARM contract rounds both products of the last minor before
            // subtracting. Newer SLP passes otherwise fuse this scalar tail,
            // unlike the historical two-lane multiply/subtract sequence.
            const float diagonal = fMat[kMScaleX] * fMat[kMScaleY];
            const float off_diagonal = fMat[kMSkewX] * fMat[kMSkewY];
            const float minor = diagonal - off_diagonal;
            output->fMat[kMPersp2] = static_cast<float>(minor * reciprocalDeterminant);
        }
#endif
    } else {
        const geometry::Affine2D inverse =
          geometry::InverseWithReciprocalDeterminant(AffinePart(*this), reciprocalDeterminant);
        output->fMat[kMScaleX] = inverse.xx;
        output->fMat[kMSkewX] = inverse.xy;
        output->fMat[kMTransX] = inverse.x_offset;
        output->fMat[kMSkewY] = inverse.yx;
        output->fMat[kMScaleY] = inverse.yy;
        output->fMat[kMTransY] = inverse.y_offset;
        output->fMat[kMPersp0] = 0.0f;
        output->fMat[kMPersp1] = 0.0f;
        output->fMat[kMPersp2] = 1.0f;
    }
    output->setTypeMask(fTypeMask);
    if (destination == this) {
        *destination = scratch;
    }
    return true;
}

}  // namespace task
}  // namespace inspirecv
