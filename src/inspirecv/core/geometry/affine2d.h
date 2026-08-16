#ifndef INSPIRECV_GEOMETRY_AFFINE2D_H_
#define INSPIRECV_GEOMETRY_AFFINE2D_H_

#include <cstddef>

namespace inspirecv {
namespace geometry {

// Backend-neutral storage and arithmetic for a row-major 2x3 affine transform:
//
//   x' = xx * x + xy * y + x_offset
//   y' = yx * x + yy * y + y_offset
//
// This type intentionally owns no backend objects. Task, OKCV, and OpenCV
// adapters can use it without introducing dependencies between those modules.
struct Affine2D {
    float xx;
    float xy;
    float x_offset;
    float yx;
    float yy;
    float y_offset;

    static Affine2D Identity() { return {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}; }
};

inline double Determinant(const Affine2D& transform) {
    return static_cast<double>(transform.xx) * transform.yy -
           static_cast<double>(transform.xy) * transform.yx;
}

inline float DeterminantFloat(const Affine2D& transform) {
    return transform.xx * transform.yy - transform.xy * transform.yx;
}

inline Affine2D InverseWithReciprocalDeterminant(const Affine2D& transform,
                                                  double reciprocal_determinant) {
    return {
      static_cast<float>(transform.yy * reciprocal_determinant),
      static_cast<float>(-transform.xy * reciprocal_determinant),
      static_cast<float>((static_cast<double>(transform.xy) * transform.y_offset -
                          static_cast<double>(transform.yy) * transform.x_offset) *
                         reciprocal_determinant),
      static_cast<float>(-transform.yx * reciprocal_determinant),
      static_cast<float>(transform.xx * reciprocal_determinant),
      static_cast<float>((static_cast<double>(transform.yx) * transform.x_offset -
                          static_cast<double>(transform.xx) * transform.y_offset) *
                         reciprocal_determinant),
    };
}

inline Affine2D InverseWithFloatDeterminant(const Affine2D& transform, float determinant) {
    return {
      transform.yy / determinant,
      -transform.xy / determinant,
      (transform.xy * transform.y_offset - transform.x_offset * transform.yy) / determinant,
      -transform.yx / determinant,
      transform.xx / determinant,
      -(transform.xx * transform.y_offset - transform.x_offset * transform.yx) / determinant,
    };
}

inline float TwoProductSum(float a, float b, float c, float d) {
    return static_cast<float>(static_cast<double>(a) * b + static_cast<double>(c) * d);
}

inline Affine2D Compose(const Affine2D& outer, const Affine2D& inner) {
    return {
      TwoProductSum(outer.xx, inner.xx, outer.xy, inner.yx),
      TwoProductSum(outer.xx, inner.xy, outer.xy, inner.yy),
      TwoProductSum(outer.xx, inner.x_offset, outer.xy, inner.y_offset) + outer.x_offset,
      TwoProductSum(outer.yx, inner.xx, outer.yy, inner.yx),
      TwoProductSum(outer.yx, inner.xy, outer.yy, inner.yy),
      TwoProductSum(outer.yx, inner.x_offset, outer.yy, inner.y_offset) + outer.y_offset,
    };
}

inline void MapPointSequential(const Affine2D& transform, float x, float y,
                               float* output_x, float* output_y) {
    *output_x = x * transform.xx + y * transform.xy + transform.x_offset;
    *output_y = x * transform.yx + y * transform.yy + transform.y_offset;
}

inline void MapPointNested(const Affine2D& transform, float x, float y,
                           float* output_x, float* output_y) {
    *output_x = x * transform.xx + (y * transform.xy + transform.x_offset);
    *output_y = x * transform.yx + (y * transform.yy + transform.y_offset);
}

}  // namespace geometry
}  // namespace inspirecv

#endif  // INSPIRECV_GEOMETRY_AFFINE2D_H_
