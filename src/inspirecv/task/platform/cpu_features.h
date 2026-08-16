#ifndef INSPIRECV_TASK_PLATFORM_CPU_FEATURES_H_
#define INSPIRECV_TASK_PLATFORM_CPU_FEATURES_H_

namespace inspirecv {
namespace task {
namespace platform {

// Return true if CPU supports SSE4.1 (x86/x86_64). False otherwise.
bool HasSse41();
// Return true if CPU supports AVX2 (x86/x86_64). False otherwise.
bool HasAvx2();

}  // namespace platform
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_PLATFORM_CPU_FEATURES_H_
