#include <inspirecv/task/geometry/gnu_projective_math.h>

namespace inspirecv {
namespace task {
namespace geometry_internal {

#if defined(__GNUC__) && !defined(__clang__)
namespace {

float DotWithStableTail(const float row[3], const float column[7]) {
    volatile const float trailingTerms =
      row[1] * column[3] + row[2] * column[6];
    return row[0] * column[0] + trailingTerms;
}

}  // namespace

void ComposeProjectiveCompat(const float left[9], const float right[9],
                             float output[9]) {
    // GCC reassociates aggregate-return expressions under -ffast-math. The
    // volatile tail retains the P0 rounding boundary on GNU toolchains.
    for (int outputRow = 0; outputRow < 3; ++outputRow) {
        const float* leftRow = left + outputRow * 3;
        for (int outputColumn = 0; outputColumn < 3; ++outputColumn) {
            output[outputRow * 3 + outputColumn] =
              DotWithStableTail(leftRow, right + outputColumn);
        }
    }
}
#endif

}  // namespace geometry_internal
}  // namespace task
}  // namespace inspirecv
