#include <inspirecv/core/geometry/affine2d.h>
#include <inspirecv/task/core/matrix.h>
#include <inspirecv/task/geometry/gnu_projective_math.h>

#include <algorithm>

namespace inspirecv {
namespace task {
namespace {

geometry::Affine2D AffinePart(const Matrix& matrix) {
    return {matrix.getScaleX(), matrix.getSkewX(), matrix.getTranslateX(),
            matrix.getSkewY(), matrix.getScaleY(), matrix.getTranslateY()};
}

bool NeedsGeneralComposition(unsigned type) {
    return (type & (Matrix::kAffine_Mask | Matrix::kPerspective_Mask)) != 0;
}

}  // namespace

bool Matrix::setRectToRect(const Rect& source, const Rect& destination, ScaleToFit alignment) {
    if (source.isEmpty()) {
        reset();
        return false;
    }
    if (destination.isEmpty()) {
        setScaleTranslate(0.0f, 0.0f, 0.0f, 0.0f);
        return true;
    }

    float horizontalScale = destination.width() / source.width();
    float verticalScale = destination.height() / source.height();
    bool extraSpaceIsHorizontal = false;
    if (alignment != kFill_ScaleToFit) {
        extraSpaceIsHorizontal = horizontalScale > verticalScale;
        const float uniformScale = std::min(horizontalScale, verticalScale);
        horizontalScale = uniformScale;
        verticalScale = uniformScale;
    }

    float horizontalOffset = destination.fLeft - source.fLeft * horizontalScale;
    float verticalOffset = destination.fTop - source.fTop * verticalScale;
    if (alignment == kCenter_ScaleToFit || alignment == kEnd_ScaleToFit) {
        float remainingSpace = extraSpaceIsHorizontal
                                 ? destination.width() - source.width() * verticalScale
                                 : destination.height() - source.height() * horizontalScale;
        if (alignment == kCenter_ScaleToFit) {
            remainingSpace *= 0.5f;
        }
        if (extraSpaceIsHorizontal) {
            horizontalOffset += remainingSpace;
        } else {
            verticalOffset += remainingSpace;
        }
    }

    setScaleTranslate(horizontalScale, verticalScale, horizontalOffset, verticalOffset);
    return true;
}

void Matrix::setConcat(const Matrix& outer, const Matrix& inner) {
    const TypeMask outerType = outer.getType();
    const TypeMask innerType = inner.getType();
    if (outer.isTriviallyIdentity()) {
        *this = inner;
        return;
    }
    if (inner.isTriviallyIdentity()) {
        *this = outer;
        return;
    }

    const unsigned combinedType = outerType | innerType;
    if (!NeedsGeneralComposition(combinedType)) {
        setScaleTranslate(outer.fMat[kMScaleX] * inner.fMat[kMScaleX],
                          outer.fMat[kMScaleY] * inner.fMat[kMScaleY],
                          outer.fMat[kMScaleX] * inner.fMat[kMTransX] + outer.fMat[kMTransX],
                          outer.fMat[kMScaleY] * inner.fMat[kMTransY] + outer.fMat[kMTransY]);
        return;
    }

    Matrix product;
    if ((combinedType & kPerspective_Mask) != 0) {
        geometry_internal::ComposeProjectiveCompat(outer.fMat, inner.fMat,
                                                   product.fMat);
        product.setTypeMask(kUnknown_Mask);
    } else {
        const geometry::Affine2D values = geometry::Compose(AffinePart(outer), AffinePart(inner));
        product.setAll(values.xx, values.xy, values.x_offset,
                       values.yx, values.yy, values.y_offset,
                       0.0f, 0.0f, 1.0f);
        product.setTypeMask(kUnknown_Mask | kOnlyPerspectiveValid_Mask);
    }
    *this = product;
}

void Matrix::preConcat(const Matrix& adjustment) {
    if (!adjustment.isIdentity()) {
        setConcat(*this, adjustment);
    }
}

void Matrix::postConcat(const Matrix& adjustment) {
    if (!adjustment.isIdentity()) {
        setConcat(adjustment, *this);
    }
}

}  // namespace task
}  // namespace inspirecv
