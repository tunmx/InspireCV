#include "../../common/common.h"

#include <cstdint>
#include <cstring>

#include "inspirecv/task/execution/row_schedule.h"

namespace {

uint32_t FloatBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void RequirePointBits(const inspirecv::task::Point& actual, float x, float y) {
    REQUIRE(FloatBits(actual.fX) == FloatBits(x));
    REQUIRE(FloatBits(actual.fY) == FloatBits(y));
}

}  // namespace

TEST_CASE("task_row_schedule_freezes_visible_span_and_coordinate_step",
          "[task][row-schedule][kernel-contract]") {
    using inspirecv::task::CLAMP_TO_EDGE;
    using inspirecv::task::Matrix;
    using inspirecv::task::ZERO;
    using inspirecv::task::internal::RowWindow;
    using inspirecv::task::internal::ScheduleRow;

    Matrix identity;
    const RowWindow direct =
      ScheduleRow(identity, identity, CLAMP_TO_EDGE, 8, 6, 2, 1, 5);
    REQUIRE(direct.first == 0);
    REQUIRE(direct.last == 5);
    RequirePointBits(direct.origin, 1.0f, 2.0f);
    RequirePointBits(direct.step, 1.0f, 0.0f);

    SECTION("left and right clipping retain legacy inclusive bounds") {
        Matrix move_left;
        move_left.setTranslate(-3.0f, 0.0f);
        Matrix move_right;
        REQUIRE(move_left.invert(&move_right));
        const RowWindow left =
          ScheduleRow(move_left, move_right, ZERO, 8, 6, 2, 0, 8);
        REQUIRE(left.first == 3);
        REQUIRE(left.last == 8);
        RequirePointBits(left.origin, 0.0f, 2.0f);
        RequirePointBits(left.step, 1.0f, 0.0f);

        move_right.setTranslate(3.0f, 0.0f);
        REQUIRE(move_right.invert(&move_left));
        const RowWindow right =
          ScheduleRow(move_right, move_left, ZERO, 8, 6, 2, 0, 8);
        REQUIRE(right.first == 0);
        REQUIRE(right.last == 5);
        RequirePointBits(right.origin, 3.0f, 2.0f);
        RequirePointBits(right.step, 1.0f, 0.0f);
    }

    SECTION("fully invisible rows collapse to an empty suffix") {
        Matrix outside;
        outside.setTranslate(-20.0f, 0.0f);
        Matrix inverse;
        REQUIRE(outside.invert(&inverse));
        const RowWindow hidden =
          ScheduleRow(outside, inverse, ZERO, 8, 6, 2, 0, 8);
        REQUIRE(hidden.first == 8);
        REQUIRE(hidden.last == 8);
        RequirePointBits(hidden.origin, -12.0f, 2.0f);
        RequirePointBits(hidden.step, 1.0f, 0.0f);
    }
}

TEST_CASE("task_row_schedule_division_preserves_exact_affine_steps",
          "[task][row-schedule][kernel-contract][regression]") {
    using namespace inspirecv::task;
    Matrix transform;
    Matrix inverse;
    for (float scale : {1.0f, 0.5f, 1.25f, -0.75f}) {
        transform.setScaleTranslate(scale, scale, 3.0f, -2.0f);
        REQUIRE(transform.invert(&inverse));
        for (int count = 1; count <= 1024; ++count) {
            const auto row = internal::ScheduleRow(
              transform, inverse, CLAMP_TO_EDGE, 2048, 2048, 7, 5, count);
            CAPTURE(scale, count);
            RequirePointBits(row.origin, 5 * scale + 3, 7 * scale - 2);
            RequirePointBits(row.step, scale, 0.0f);
            REQUIRE(row.first == 0);
            REQUIRE(row.last == count);
        }
    }
}
