#include "cv_transform_matrix.h"
#include "check.h"
#include <inspirecv/core/geometry/affine2d.h>
namespace okcv {

TransformMatrix::TransformMatrix(const std::initializer_list<float> &v) {
    INSPIRECV_CHECK_EQ(v.size(), 6);
    int i = 0;
    for (const auto &d : v)
        data_[i++] = d;
}

TransformMatrix TransformMatrix::Inv() const {
    const inspirecv::geometry::Affine2D source = {
      data_[0], data_[1], data_[2], data_[3], data_[4], data_[5]};
    const float det = inspirecv::geometry::DeterminantFloat(source);
    INSPIRECV_CHECK_NE(det, 0) << *this;
    const inspirecv::geometry::Affine2D result =
      inspirecv::geometry::InverseWithFloatDeterminant(source, det);
    TransformMatrix inv;
    inv[0] = result.xx;
    inv[1] = result.xy;
    inv[2] = result.x_offset;
    inv[3] = result.yx;
    inv[4] = result.yy;
    inv[5] = result.y_offset;
    return inv;
}

bool TransformMatrix::IsIdentity(float eps) const {
    return IsNearlyEqual(data_[0], 1.f, eps) && IsNearlyEqual(data_[1], 0.f, eps) &&
           IsNearlyEqual(data_[2], 0.f, eps) && IsNearlyEqual(data_[3], 0.f, eps) &&
           IsNearlyEqual(data_[4], 1.f, eps) && IsNearlyEqual(data_[5], 0.f, eps);
}

bool TransformMatrix::IsCrop(float eps) const {
    return IsNearlyEqual(data_[0], 1.f, eps) && IsNearlyEqual(data_[1], 0.f, eps) &&
           IsNearlyEqual(data_[2], 0.f, eps) && IsNearlyEqual(data_[4], 1.f, eps);
}

bool TransformMatrix::IsResize(float eps) const {
    return IsNearlyEqual(data_[1], 0.f, eps) && IsNearlyEqual(data_[2], 0.f, eps) &&
           IsNearlyEqual(data_[3], 0.f, eps) && IsNearlyEqual(data_[5], 0.f, eps);
}

bool TransformMatrix::IsCropAndResize(float eps) const {
    return IsNearlyEqual(data_[1], 0.f, eps) && IsNearlyEqual(data_[3], 0.f, eps);
}

}  // namespace okcv
