#ifndef INSPIRECV_TASK_PLATFORM_CPU_FEATURES_H_
#define INSPIRECV_TASK_PLATFORM_CPU_FEATURES_H_

// MSVC exposes SSE4.1 intrinsics on x86 without defining __SSE4_1__.
// This describes compiled helpers only; callers still check runtime CPUID.
#if defined(INSPIRECV_TASK_USE_SSE) && \
    (defined(__SSE4_1__) || \
     (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))))
#define INSPIRECV_TASK_HAVE_SSE41_INTRINSICS
#endif

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
