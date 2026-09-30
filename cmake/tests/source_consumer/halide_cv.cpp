#include <inspirecv/task/core/st_types.h>

#include <cstring>

static_assert(sizeof(halide_type_t) == sizeof(uint32_t), "Halide type layout must remain stable");

extern "C" __declspec(dllexport) uint32_t InspireCvHalideType() {
    const auto type = halide_type_of<float>();
    uint32_t bits = 0;
    std::memcpy(&bits, &type, sizeof(bits));
    return bits;
}
