#include <inspirecv/task/core/matrix.h>

#include <cstdint>
#include <cstring>

namespace inspirecv {
namespace task {
namespace {

int32_t OrderedMagnitudeBits(float value) {
    int32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if (bits >= 0) {
        return bits;
    }
    return -(bits & INT32_C(0x7fffffff));
}

}  // namespace

uint8_t Matrix::computeTypeMask() const {
    if (fMat[kMPersp0] != 0.0f || fMat[kMPersp1] != 0.0f || fMat[kMPersp2] != 1.0f) {
        return static_cast<uint8_t>(kORableMasks);
    }

    unsigned classification =
      (fMat[kMTransX] != 0.0f || fMat[kMTransY] != 0.0f) ? kTranslate_Mask : 0;
    const int32_t scaleXBits = OrderedMagnitudeBits(fMat[kMScaleX]);
    const int32_t scaleYBits = OrderedMagnitudeBits(fMat[kMScaleY]);
    const int32_t skewXBits = OrderedMagnitudeBits(fMat[kMSkewX]);
    const int32_t skewYBits = OrderedMagnitudeBits(fMat[kMSkewY]);

    if (skewXBits != 0 || skewYBits != 0) {
        classification |= kAffine_Mask | kScale_Mask;
        const bool swapsAxes = (scaleXBits | scaleYBits) == 0 &&
                               skewXBits != 0 && skewYBits != 0;
        if (swapsAxes) {
            classification |= kRectStaysRect_Mask;
        }
    } else {
        if (scaleXBits != INT32_C(0x3f800000) ||
            scaleYBits != INT32_C(0x3f800000)) {
            classification |= kScale_Mask;
        }
        if (scaleXBits != 0 && scaleYBits != 0) {
            classification |= kRectStaysRect_Mask;
        }
    }
    return static_cast<uint8_t>(classification);
}

uint8_t Matrix::computePerspectiveTypeMask() const {
    const bool hasPerspective = fMat[kMPersp0] != 0.0f || fMat[kMPersp1] != 0.0f ||
                                fMat[kMPersp2] != 1.0f;
    return static_cast<uint8_t>(hasPerspective
                                  ? kORableMasks
                                  : kOnlyPerspectiveValid_Mask | kUnknown_Mask);
}

}  // namespace task
}  // namespace inspirecv
