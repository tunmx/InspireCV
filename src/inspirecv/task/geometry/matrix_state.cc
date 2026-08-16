#include <inspirecv/task/core/matrix.h>

#include <algorithm>
#include <cstring>

namespace inspirecv {
namespace task {
namespace {

constexpr float kIdentityValues[9] = {
  1.0f, 0.0f, 0.0f,
  0.0f, 1.0f, 0.0f,
  0.0f, 0.0f, 1.0f,
};

constexpr int kAffineSlots[6] = {
  Matrix::kMScaleX,
  Matrix::kMSkewY,
  Matrix::kMSkewX,
  Matrix::kMScaleY,
  Matrix::kMTransX,
  Matrix::kMTransY,
};

}  // namespace

void Matrix::reset() {
    std::memcpy(fMat, kIdentityValues, sizeof(fMat));
    setTypeMask(kIdentity_Mask | kRectStaysRect_Mask);
}

void Matrix::set9(const float values[]) {
    std::memcpy(fMat, values, sizeof(fMat));
    setTypeMask(kUnknown_Mask);
}

bool operator==(const Matrix& left, const Matrix& right) {
    return std::equal(left.fMat, left.fMat + 9, right.fMat);
}

void Matrix::SetAffineIdentity(float affine[6]) {
    const float identity[6] = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    std::copy(identity, identity + 6, affine);
}

bool Matrix::asAffine(float affine[6]) const {
    if (affine != nullptr) {
        for (int index = 0; index < 6; ++index) {
            affine[index] = fMat[kAffineSlots[index]];
        }
    }
    return true;
}

void Matrix::setAffine(const float affine[6]) {
    for (int index = 0; index < 6; ++index) {
        fMat[kAffineSlots[index]] = affine[index];
    }
    fMat[kMPersp0] = 0.0f;
    fMat[kMPersp1] = 0.0f;
    fMat[kMPersp2] = 1.0f;
    setTypeMask(kUnknown_Mask | kOnlyPerspectiveValid_Mask);
}

}  // namespace task
}  // namespace inspirecv
