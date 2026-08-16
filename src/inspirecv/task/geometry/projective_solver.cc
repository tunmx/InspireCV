#include <inspirecv/task/geometry/projective_solver.h>

#include <inspirecv/task/core/matrix.h>

namespace inspirecv {
namespace task {
namespace geometry_internal {
namespace {

bool IsZeroAfterUnderflow(float value) {
    return value * value == 0.0f;
}

bool PreferHorizontalDivision(float horizontal, float vertical) {
    if (horizontal > 0.0f) {
        return vertical > 0.0f ? horizontal > vertical : horizontal > -vertical;
    }
    return vertical > 0.0f ? -horizontal > vertical : horizontal < vertical;
}

void StoreTwoPointBasis(const Point* points, float output[9]) {
    const float deltaX = points[1].fX - points[0].fX;
    const float deltaY = points[1].fY - points[0].fY;
    output[Matrix::kMScaleX] = deltaY;
    output[Matrix::kMSkewX] = deltaX;
    output[Matrix::kMTransX] = points[0].fX;
    output[Matrix::kMSkewY] = points[0].fX - points[1].fX;
    output[Matrix::kMScaleY] = deltaY;
    output[Matrix::kMTransY] = points[0].fY;
    output[Matrix::kMPersp0] = 0.0f;
    output[Matrix::kMPersp1] = 0.0f;
    output[Matrix::kMPersp2] = 1.0f;
}

void StoreThreePointBasis(const Point* points, float output[9]) {
    const Point& origin = points[0];
    output[Matrix::kMScaleX] = points[2].fX - origin.fX;
    output[Matrix::kMSkewX] = points[1].fX - origin.fX;
    output[Matrix::kMTransX] = origin.fX;
    output[Matrix::kMSkewY] = points[2].fY - origin.fY;
    output[Matrix::kMScaleY] = points[1].fY - origin.fY;
    output[Matrix::kMTransY] = origin.fY;
    output[Matrix::kMPersp0] = 0.0f;
    output[Matrix::kMPersp1] = 0.0f;
    output[Matrix::kMPersp2] = 1.0f;
}

bool SolveFirstPerspectiveWeight(float diagonalX, float diagonalY,
                                 float side1X, float side1Y,
                                 float side2X, float side2Y,
                                 float* weight) {
    if (PreferHorizontalDivision(side2X, side2Y)) {
        const float denominator = side1X * side2Y / side2X - side1Y;
        if (IsZeroAfterUnderflow(denominator)) {
            return false;
        }
        *weight = (((diagonalX - side1X) * side2Y / side2X) -
                   diagonalY + side1Y) / denominator;
    } else {
        const float denominator = side1X - side1Y * side2X / side2Y;
        if (IsZeroAfterUnderflow(denominator)) {
            return false;
        }
        *weight = (diagonalX - side1X -
                   (diagonalY - side1Y) * side2X / side2Y) / denominator;
    }
    return true;
}

bool SolveSecondPerspectiveWeight(float diagonalX, float diagonalY,
                                  float side1X, float side1Y,
                                  float side2X, float side2Y,
                                  float* weight) {
    if (PreferHorizontalDivision(side1X, side1Y)) {
        const float denominator = side2Y - side2X * side1Y / side1X;
        if (IsZeroAfterUnderflow(denominator)) {
            return false;
        }
        *weight = (diagonalY - side2Y -
                   (diagonalX - side2X) * side1Y / side1X) / denominator;
    } else {
        const float denominator = side2Y * side1X / side1Y - side2X;
        if (IsZeroAfterUnderflow(denominator)) {
            return false;
        }
        *weight = ((diagonalY - side2Y) * side1X / side1Y -
                   diagonalX + side2X) / denominator;
    }
    return true;
}

bool StoreFourPointBasis(const Point* points, float output[9]) {
    const float diagonalX = points[2].fX - points[0].fX;
    const float diagonalY = points[2].fY - points[0].fY;
    const float side1X = points[2].fX - points[1].fX;
    const float side1Y = points[2].fY - points[1].fY;
    const float side2X = points[2].fX - points[3].fX;
    const float side2Y = points[2].fY - points[3].fY;

    float firstWeight;
    float secondWeight;
    if (!SolveFirstPerspectiveWeight(diagonalX, diagonalY, side1X, side1Y,
                                     side2X, side2Y, &firstWeight) ||
        !SolveSecondPerspectiveWeight(diagonalX, diagonalY, side1X, side1Y,
                                      side2X, side2Y, &secondWeight)) {
        return false;
    }

    const Point& origin = points[0];
    output[Matrix::kMScaleX] =
      secondWeight * points[3].fX + points[3].fX - origin.fX;
    output[Matrix::kMSkewX] =
      firstWeight * points[1].fX + points[1].fX - origin.fX;
    output[Matrix::kMTransX] = origin.fX;
    output[Matrix::kMSkewY] =
      secondWeight * points[3].fY + points[3].fY - origin.fY;
    output[Matrix::kMScaleY] =
      firstWeight * points[1].fY + points[1].fY - origin.fY;
    output[Matrix::kMTransY] = origin.fY;
    output[Matrix::kMPersp0] = secondWeight;
    output[Matrix::kMPersp1] = firstWeight;
    output[Matrix::kMPersp2] = 1.0f;
    return true;
}

}  // namespace

bool BuildPointBasis(const Point* points, int count, float output[9]) {
    if (count == 2) {
        StoreTwoPointBasis(points, output);
        return true;
    }
    if (count == 3) {
        StoreThreePointBasis(points, output);
        return true;
    }
    return count == 4 && StoreFourPointBasis(points, output);
}

}  // namespace geometry_internal
}  // namespace task
}  // namespace inspirecv
