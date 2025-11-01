#ifndef INSPIRECV_STREAMTASK_CORE_CPU_FEATURES_H_
#define INSPIRECV_STREAMTASK_CORE_CPU_FEATURES_H_

namespace inspirecv {

// Return true if CPU supports SSE4.1 (x86/x86_64). False otherwise.
bool cpu_has_sse41();
// Return true if CPU supports AVX2 (x86/x86_64). False otherwise.
bool cpu_has_avx2();

}

#endif // INSPIRECV_STREAMTASK_CORE_CPU_FEATURES_H_




