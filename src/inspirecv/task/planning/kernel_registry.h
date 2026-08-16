#ifndef INSPIRECV_TASK_PLANNING_KERNEL_REGISTRY_H_
#define INSPIRECV_TASK_PLANNING_KERNEL_REGISTRY_H_

#include "inspirecv/task/planning/pixel_program.h"

namespace inspirecv {
namespace task {
namespace internal {

ConvertSpan FindColorConverter(StreamFormat source, StreamFormat destination);
ConvertSpan FindDrawWriter(int pixel_bytes);
WriteInterleavedSpan FindInterleavedFloatWriter(StreamFormat format, int destination_channels);
WritePlanarSpan FindPlanarFloatWriter(StreamFormat format, int destination_channels);
SampleSpan FindSampler(StreamFormat format, Filter filter, bool direct_sampling);

}  // namespace internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_PLANNING_KERNEL_REGISTRY_H_
