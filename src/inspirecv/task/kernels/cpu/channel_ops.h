#ifndef INSPIRECV_TASK_KERNELS_CPU_CHANNEL_OPS_H_
#define INSPIRECV_TASK_KERNELS_CPU_CHANNEL_OPS_H_

#include <cstddef>
#include <cstdint>

namespace inspirecv {
namespace task {
namespace kernels {
namespace channel {

using PixelTransform = void(const uint8_t*, uint8_t*, size_t);

PixelTransform CopyMonoPixels;
PixelTransform CopyTriplePixels;
PixelTransform CopyQuadPixels;
PixelTransform ReplicateMonoToTriple;
PixelTransform ReplicateMonoToQuad;
PixelTransform AppendOpaqueAlpha;
PixelTransform ReverseTriple;
PixelTransform ReverseQuadColor;
PixelTransform DropAlpha;
PixelTransform ReverseAndDropAlpha;
PixelTransform LumaFromRgb;
PixelTransform LumaFromBgr;
PixelTransform LumaFromRgba;
PixelTransform LumaFromBgra;
PixelTransform FillMonoPixels;
PixelTransform FillTriplePixels;
PixelTransform FillQuadPixels;

}  // namespace channel
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_KERNELS_CPU_CHANNEL_OPS_H_
