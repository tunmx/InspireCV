#ifndef INSPIRECV_TASK_EXECUTION_ROW_SCHEDULE_H_
#define INSPIRECV_TASK_EXECUTION_ROW_SCHEDULE_H_

#include <inspirecv/task/core/matrix.h>
#include <inspirecv/task/core/preprocess_types.h>

namespace inspirecv {
namespace task {
namespace internal {

struct RowWindow {
    Point origin = {0.0f, 0.0f};
    Point step = {0.0f, 0.0f};
    int first = 0;
    int last = 0;
};

RowWindow ScheduleRow(const Matrix& destination_to_source,
                      const Matrix& source_to_destination, Wrap wrap,
                      int source_width, int source_height, int destination_y,
                      int destination_x, int pixel_count);

}  // namespace internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_EXECUTION_ROW_SCHEDULE_H_
