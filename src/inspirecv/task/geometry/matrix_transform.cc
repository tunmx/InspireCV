#include <inspirecv/task/core/matrix.h>

#include <cmath>

namespace inspirecv {
namespace task {
namespace {

constexpr float kDegreesToRadians = 3.14159265f / 180.0f;

inline float PairSum(float firstValue, float firstWeight,
                     float secondValue, float secondWeight) {
    return firstValue * firstWeight + secondValue * secondWeight;
}

inline bool IsUnitScale(float horizontal, float vertical) {
    return horizontal == 1.0f && vertical == 1.0f;
}

}  // namespace

void Matrix::setTranslate(float horizontal, float vertical) {
    if (horizontal == 0.0f && vertical == 0.0f) {
        reset();
        return;
    }
    setScaleTranslate(1.0f, 1.0f, horizontal, vertical);
}

void Matrix::preTranslate(float horizontal, float vertical) {
    const TypeMask type = getType();
    if ((type & kPerspective_Mask) != 0) {
        Matrix adjustment;
        adjustment.setTranslate(horizontal, vertical);
        preConcat(adjustment);
        return;
    }

    float deltaX = horizontal;
    float deltaY = vertical;
    if (type > kTranslate_Mask) {
        deltaX = PairSum(fMat[kMScaleX], horizontal, fMat[kMSkewX], vertical);
        deltaY = PairSum(fMat[kMSkewY], horizontal, fMat[kMScaleY], vertical);
    }
    fMat[kMTransX] += deltaX;
    fMat[kMTransY] += deltaY;
    updateTranslateMask();
}

void Matrix::postTranslate(float horizontal, float vertical) {
    Matrix adjustment;
    adjustment.setTranslate(horizontal, vertical);
    postConcat(adjustment);
}

void Matrix::setScale(float horizontal, float vertical, float pivotX, float pivotY) {
    if (IsUnitScale(horizontal, vertical)) {
        reset();
        return;
    }
    setScaleTranslate(horizontal, vertical,
                      pivotX - horizontal * pivotX,
                      pivotY - vertical * pivotY);
}

void Matrix::setScale(float horizontal, float vertical) {
    if (IsUnitScale(horizontal, vertical)) {
        reset();
        return;
    }
    setScaleTranslate(horizontal, vertical, 0.0f, 0.0f);
}

void Matrix::preScale(float horizontal, float vertical, float pivotX, float pivotY) {
    if (IsUnitScale(horizontal, vertical)) {
        return;
    }
    Matrix adjustment;
    adjustment.setScale(horizontal, vertical, pivotX, pivotY);
    preConcat(adjustment);
}

void Matrix::preScale(float horizontal, float vertical) {
    if (IsUnitScale(horizontal, vertical)) {
        return;
    }

    fMat[kMScaleX] *= horizontal;
    fMat[kMSkewY] *= horizontal;
    fMat[kMPersp0] *= horizontal;
    fMat[kMSkewX] *= vertical;
    fMat[kMScaleY] *= vertical;
    fMat[kMPersp1] *= vertical;

    const bool remainsUnitScale = fMat[kMScaleX] == 1.0f && fMat[kMScaleY] == 1.0f;
    const bool hasComplexTerms = (fTypeMask & (kPerspective_Mask | kAffine_Mask)) != 0;
    if (remainsUnitScale && !hasComplexTerms) {
        clearTypeMask(kScale_Mask);
    } else {
        orTypeMask(kScale_Mask);
    }
}

void Matrix::postScale(float horizontal, float vertical, float pivotX, float pivotY) {
    if (IsUnitScale(horizontal, vertical)) {
        return;
    }
    Matrix adjustment;
    adjustment.setScale(horizontal, vertical, pivotX, pivotY);
    postConcat(adjustment);
}

void Matrix::postScale(float horizontal, float vertical) {
    if (IsUnitScale(horizontal, vertical)) {
        return;
    }
    Matrix adjustment;
    adjustment.setScale(horizontal, vertical);
    postConcat(adjustment);
}

bool Matrix::postIDiv(int horizontalDivisor, int verticalDivisor) {
    if (horizontalDivisor == 0 || verticalDivisor == 0) {
        return false;
    }
    const float horizontalFactor = 1.0f / horizontalDivisor;
    const float verticalFactor = 1.0f / verticalDivisor;
    for (int index = kMScaleX; index <= kMTransX; ++index) {
        fMat[index] *= horizontalFactor;
    }
    for (int index = kMSkewY; index <= kMTransY; ++index) {
        fMat[index] *= verticalFactor;
    }
    setTypeMask(kUnknown_Mask);
    return true;
}

void Matrix::setSinCos(float sine, float cosine, float pivotX, float pivotY) {
    const float cosineDelta = 1.0f - cosine;
    setAll(cosine, -sine,
           PairSum(sine, pivotY, cosineDelta, pivotX),
           sine, cosine,
           PairSum(-sine, pivotX, cosineDelta, pivotY),
           0.0f, 0.0f, 1.0f);
    setTypeMask(kUnknown_Mask | kOnlyPerspectiveValid_Mask);
}

void Matrix::setSinCos(float sine, float cosine) {
    setAll(cosine, -sine, 0.0f,
           sine, cosine, 0.0f,
           0.0f, 0.0f, 1.0f);
    setTypeMask(kUnknown_Mask | kOnlyPerspectiveValid_Mask);
}

void Matrix::setRotate(float degrees, float pivotX, float pivotY) {
    const float radians = degrees * kDegreesToRadians;
    setSinCos(sin(radians), cos(radians), pivotX, pivotY);
}

void Matrix::setRotate(float degrees) {
    const float radians = degrees * kDegreesToRadians;
    setSinCos(sin(radians), cos(radians));
}

void Matrix::preRotate(float degrees, float pivotX, float pivotY) {
    Matrix adjustment;
    adjustment.setRotate(degrees, pivotX, pivotY);
    preConcat(adjustment);
}

void Matrix::preRotate(float degrees) {
    Matrix adjustment;
    adjustment.setRotate(degrees);
    preConcat(adjustment);
}

void Matrix::postRotate(float degrees, float pivotX, float pivotY) {
    Matrix adjustment;
    adjustment.setRotate(degrees, pivotX, pivotY);
    postConcat(adjustment);
}

void Matrix::postRotate(float degrees) {
    Matrix adjustment;
    adjustment.setRotate(degrees);
    postConcat(adjustment);
}

void Matrix::setSkew(float horizontal, float vertical, float pivotX, float pivotY) {
    setAll(1.0f, horizontal, -horizontal * pivotY,
           vertical, 1.0f, -vertical * pivotX,
           0.0f, 0.0f, 1.0f);
    setTypeMask(kUnknown_Mask | kOnlyPerspectiveValid_Mask);
}

void Matrix::setSkew(float horizontal, float vertical) {
    setAll(1.0f, horizontal, 0.0f,
           vertical, 1.0f, 0.0f,
           0.0f, 0.0f, 1.0f);
    setTypeMask(kUnknown_Mask | kOnlyPerspectiveValid_Mask);
}

void Matrix::preSkew(float horizontal, float vertical, float pivotX, float pivotY) {
    Matrix adjustment;
    adjustment.setSkew(horizontal, vertical, pivotX, pivotY);
    preConcat(adjustment);
}

void Matrix::preSkew(float horizontal, float vertical) {
    Matrix adjustment;
    adjustment.setSkew(horizontal, vertical);
    preConcat(adjustment);
}

void Matrix::postSkew(float horizontal, float vertical, float pivotX, float pivotY) {
    Matrix adjustment;
    adjustment.setSkew(horizontal, vertical, pivotX, pivotY);
    postConcat(adjustment);
}

void Matrix::postSkew(float horizontal, float vertical) {
    Matrix adjustment;
    adjustment.setSkew(horizontal, vertical);
    postConcat(adjustment);
}

}  // namespace task
}  // namespace inspirecv
