#include <cstdint>
#include <cstring>

// Model the host's non-POD Halide type (as used by MNN) in a separate TU.
// Its same-named helper has a different MSVC return ABI from CV's POD type.
struct halide_type_t {
    __declspec(align(1)) uint8_t code;
    __declspec(align(1)) uint8_t bits;
    __declspec(align(2)) uint16_t lanes;

    halide_type_t(uint8_t code, uint8_t bits, uint16_t lanes = 1)
        : code(code), bits(bits), lanes(lanes) {}
    halide_type_t() : code(0), bits(0), lanes(0) {}
};

static_assert(sizeof(halide_type_t) == sizeof(uint32_t), "Host Halide type must be four bytes");

template <typename T>
inline halide_type_t halide_type_of();

template <>
inline halide_type_t halide_type_of<float>() {
    return halide_type_t(2, 32);
}

extern "C" __declspec(dllexport) uint32_t HostHalideType() {
    // Assignment also exercises the hidden return pointer used by a non-POD
    // result; direct initialization can accidentally read a preceding call's
    // stack value when the linker selects the incompatible POD helper.
    halide_type_t type;
    type = halide_type_of<float>();
    uint32_t bits = 0;
    std::memcpy(&bits, &type, sizeof(bits));
    return bits;
}
