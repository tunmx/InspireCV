#ifndef INSPIRECV_GEOMETRY_PROJECTIVE2D_H_
#define INSPIRECV_GEOMETRY_PROJECTIVE2D_H_

namespace inspirecv {
namespace geometry {

// Backend-neutral row-major 3x3 transform. Affine2D remains the preferred
// representation for hot affine paths; this type exists for compatibility
// operations that genuinely require a projective third row.
struct Projective2D {
    float xx;
    float xy;
    float xw;
    float yx;
    float yy;
    float yw;
    float wx;
    float wy;
    float ww;
};

inline Projective2D LoadProjective(const float values[9]) {
    return {values[0], values[1], values[2],
            values[3], values[4], values[5],
            values[6], values[7], values[8]};
}

inline void StoreProjective(const Projective2D& transform, float values[9]) {
    values[0] = transform.xx;
    values[1] = transform.xy;
    values[2] = transform.xw;
    values[3] = transform.yx;
    values[4] = transform.yy;
    values[5] = transform.yw;
    values[6] = transform.wx;
    values[7] = transform.wy;
    values[8] = transform.ww;
}

inline float RowColumn(float left0, float left1, float left2,
                       float right0, float right1, float right2) {
    return left0 * right0 + left1 * right1 + left2 * right2;
}

inline Projective2D Compose(const Projective2D& outer, const Projective2D& inner) {
    return {
      RowColumn(outer.xx, outer.xy, outer.xw, inner.xx, inner.yx, inner.wx),
      RowColumn(outer.xx, outer.xy, outer.xw, inner.xy, inner.yy, inner.wy),
      RowColumn(outer.xx, outer.xy, outer.xw, inner.xw, inner.yw, inner.ww),
      RowColumn(outer.yx, outer.yy, outer.yw, inner.xx, inner.yx, inner.wx),
      RowColumn(outer.yx, outer.yy, outer.yw, inner.xy, inner.yy, inner.wy),
      RowColumn(outer.yx, outer.yy, outer.yw, inner.xw, inner.yw, inner.ww),
      RowColumn(outer.wx, outer.wy, outer.ww, inner.xx, inner.yx, inner.wx),
      RowColumn(outer.wx, outer.wy, outer.ww, inner.xy, inner.yy, inner.wy),
      RowColumn(outer.wx, outer.wy, outer.ww, inner.xw, inner.yw, inner.ww),
    };
}

inline float RowColumn(const float row[3], const float column[7]) {
    return row[0] * column[0] + row[1] * column[3] + row[2] * column[6];
}

inline Projective2D Compose(const float outer[9], const float inner[9]) {
    return {
      RowColumn(&outer[0], &inner[0]),
      RowColumn(&outer[0], &inner[1]),
      RowColumn(&outer[0], &inner[2]),
      RowColumn(&outer[3], &inner[0]),
      RowColumn(&outer[3], &inner[1]),
      RowColumn(&outer[3], &inner[2]),
      RowColumn(&outer[6], &inner[0]),
      RowColumn(&outer[6], &inner[1]),
      RowColumn(&outer[6], &inner[2]),
    };
}

inline double DoubleCross(double firstA, double firstB,
                          double secondA, double secondB) {
    return firstA * firstB - secondA * secondB;
}

inline double ReciprocalDeterminant(const Projective2D& transform, bool perspective) {
    double determinant;
    if (perspective) {
        determinant =
          transform.xx * DoubleCross(transform.yy, transform.ww,
                                     transform.yw, transform.wy) +
          transform.xy * DoubleCross(transform.yw, transform.wx,
                                     transform.yx, transform.ww) +
          transform.xw * DoubleCross(transform.yx, transform.wy,
                                     transform.yy, transform.wx);
    } else {
        determinant = DoubleCross(transform.xx, transform.yy,
                                  transform.xy, transform.yx);
    }
    return 1.0 / determinant;
}

inline double ReciprocalDeterminant(const float transform[9], bool perspective) {
    double determinant;
    if (perspective) {
        determinant =
          transform[0] * DoubleCross(transform[4], transform[8],
                                     transform[5], transform[7]) +
          transform[1] * DoubleCross(transform[5], transform[6],
                                     transform[3], transform[8]) +
          transform[2] * DoubleCross(transform[3], transform[7],
                                     transform[4], transform[6]);
    } else {
        determinant = DoubleCross(transform[0], transform[4],
                                  transform[1], transform[3]);
    }
    return 1.0 / determinant;
}

inline float FloatMinorScaled(float firstA, float firstB,
                              float secondA, float secondB, double scale) {
    const float minor = firstA * firstB - secondA * secondB;
    return static_cast<float>(minor * scale);
}

inline Projective2D InverseProjective(const Projective2D& source,
                                      double reciprocal_determinant) {
    return {
      FloatMinorScaled(source.yy, source.ww, source.yw, source.wy,
                       reciprocal_determinant),
      FloatMinorScaled(source.xw, source.wy, source.xy, source.ww,
                       reciprocal_determinant),
      FloatMinorScaled(source.xy, source.yw, source.xw, source.yy,
                       reciprocal_determinant),
      FloatMinorScaled(source.yw, source.wx, source.yx, source.ww,
                       reciprocal_determinant),
      FloatMinorScaled(source.xx, source.ww, source.xw, source.wx,
                       reciprocal_determinant),
      FloatMinorScaled(source.xw, source.yx, source.xx, source.yw,
                       reciprocal_determinant),
      FloatMinorScaled(source.yx, source.wy, source.yy, source.wx,
                       reciprocal_determinant),
      FloatMinorScaled(source.xy, source.wx, source.xx, source.wy,
                       reciprocal_determinant),
      FloatMinorScaled(source.xx, source.yy, source.xy, source.yx,
                       reciprocal_determinant),
    };
}

inline Projective2D InverseProjective(const float source[9],
                                      double reciprocal_determinant) {
    return {
      FloatMinorScaled(source[4], source[8], source[5], source[7],
                       reciprocal_determinant),
      FloatMinorScaled(source[2], source[7], source[1], source[8],
                       reciprocal_determinant),
      FloatMinorScaled(source[1], source[5], source[2], source[4],
                       reciprocal_determinant),
      FloatMinorScaled(source[5], source[6], source[3], source[8],
                       reciprocal_determinant),
      FloatMinorScaled(source[0], source[8], source[2], source[6],
                       reciprocal_determinant),
      FloatMinorScaled(source[2], source[3], source[0], source[5],
                       reciprocal_determinant),
      FloatMinorScaled(source[3], source[7], source[4], source[6],
                       reciprocal_determinant),
      FloatMinorScaled(source[1], source[6], source[0], source[7],
                       reciprocal_determinant),
      FloatMinorScaled(source[0], source[4], source[1], source[3],
                       reciprocal_determinant),
    };
}

inline void StoreInverseProjective(const float source[9],
                                   double reciprocal_determinant,
                                   float destination[9]) {
    destination[0] = FloatMinorScaled(source[4], source[8], source[5], source[7],
                                      reciprocal_determinant);
    destination[1] = FloatMinorScaled(source[2], source[7], source[1], source[8],
                                      reciprocal_determinant);
    destination[2] = FloatMinorScaled(source[1], source[5], source[2], source[4],
                                      reciprocal_determinant);
    destination[3] = FloatMinorScaled(source[5], source[6], source[3], source[8],
                                      reciprocal_determinant);
    destination[4] = FloatMinorScaled(source[0], source[8], source[2], source[6],
                                      reciprocal_determinant);
    destination[5] = FloatMinorScaled(source[2], source[3], source[0], source[5],
                                      reciprocal_determinant);
    destination[6] = FloatMinorScaled(source[3], source[7], source[4], source[6],
                                      reciprocal_determinant);
    destination[7] = FloatMinorScaled(source[1], source[6], source[0], source[7],
                                      reciprocal_determinant);
    destination[8] = FloatMinorScaled(source[0], source[4], source[1], source[3],
                                      reciprocal_determinant);
}

}  // namespace geometry
}  // namespace inspirecv

#endif  // INSPIRECV_GEOMETRY_PROJECTIVE2D_H_
