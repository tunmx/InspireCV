#include "inspirecv/task/execution/row_schedule.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace inspirecv {
namespace task {
namespace internal {
namespace {

enum Boundary : uint8_t {
    kOutsideLeft = 1 << 0,
    kOutsideRight = 1 << 1,
    kOutsideTop = 1 << 2,
    kOutsideBottom = 1 << 3,
};

uint8_t Classify(const Point& point, int width, int height) {
    uint8_t result = 0;
    if (point.fX < 0.0f) result |= kOutsideLeft;
    if (point.fX > static_cast<float>(width - 1)) result |= kOutsideRight;
    if (point.fY < 0.0f) result |= kOutsideTop;
    if (point.fY > static_cast<float>(height - 1)) result |= kOutsideBottom;
    return result;
}

void MoveToBoundary(Point* point, uint8_t boundary, float y_per_x,
                    float x_per_y, int width, int height) {
    if ((boundary & kOutsideLeft) != 0) {
        point->fY += y_per_x * (0.0f - point->fX);
        point->fX = 0.0f;
    } else if ((boundary & kOutsideRight) != 0) {
        point->fY += y_per_x * (static_cast<float>(width - 1) - point->fX);
        point->fX = static_cast<float>(width - 1);
    } else if ((boundary & kOutsideBottom) != 0) {
        point->fX += x_per_y * (static_cast<float>(height - 1) - point->fY);
        point->fY = static_cast<float>(height - 1);
    } else if ((boundary & kOutsideTop) != 0) {
        point->fX += x_per_y * (0.0f - point->fY);
        point->fY = 0.0f;
    }
}

void ClipVisibleRange(Point endpoints[2], const Matrix& source_to_destination,
                      int source_width, int source_height, int destination_x,
                      int pixel_count, int* first, int* last) {
    uint8_t outside[2] = {Classify(endpoints[0], source_width, source_height),
                          Classify(endpoints[1], source_width, source_height)};
    const float change_x = endpoints[1].fX - endpoints[0].fX;
    const float change_y = endpoints[1].fY - endpoints[0].fY;
    const float y_per_x = std::fabs(change_x) > 0.01f ? change_y / change_x : 0.0f;
    const float x_per_y = std::fabs(y_per_x) > 0.01f ? change_x / change_y : 0.0f;

    *first = 0;
    *last = pixel_count;
    while (outside[0] != 0 || outside[1] != 0) {
        if ((outside[0] & outside[1]) != 0) {
            *first = *last;
            break;
        }
        const int endpoint = outside[0] != 0 ? 0 : 1;
        MoveToBoundary(&endpoints[endpoint], outside[endpoint], y_per_x, x_per_y,
                       source_width, source_height);
        const Point destination = source_to_destination.mapXY(
          endpoints[endpoint].fX, endpoints[endpoint].fY);
        outside[endpoint] =
          Classify(endpoints[endpoint], source_width, source_height);
        if (endpoint == 0) {
            *first = static_cast<int>(::round(destination.fX)) - destination_x;
        } else {
            *last = static_cast<int>(::floor(destination.fX)) - destination_x + 1;
        }
    }
    *last = std::min(*last, pixel_count);
    *first = std::min(*first, *last);
}

}  // namespace

RowWindow ScheduleRow(const Matrix& destination_to_source,
                      const Matrix& source_to_destination, Wrap wrap,
                      int source_width, int source_height, int destination_y,
                      int destination_x, int pixel_count) {
    Point endpoints[2] = {
      {static_cast<float>(destination_x), static_cast<float>(destination_y)},
      {static_cast<float>(destination_x + pixel_count),
       static_cast<float>(destination_y)}};
    destination_to_source.mapPoints(endpoints, 2);
    const Point change = {endpoints[1].fX - endpoints[0].fX,
                          endpoints[1].fY - endpoints[0].fY};

    RowWindow window;
    window.first = 0;
    window.last = pixel_count;
    if (wrap == ZERO) {
        ClipVisibleRange(endpoints, source_to_destination, source_width,
                         source_height, destination_x, pixel_count,
                         &window.first, &window.last);
        window.origin = {static_cast<float>(destination_x + window.first),
                         static_cast<float>(destination_y)};
        destination_to_source.mapPoints(&window.origin, 1);
    } else {
        window.origin = endpoints[0];
    }
    window.step = {change.fX / static_cast<float>(pixel_count),
                   change.fY / static_cast<float>(pixel_count)};
    return window;
}

}  // namespace internal
}  // namespace task
}  // namespace inspirecv
