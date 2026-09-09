#include <inspirecv/core/geometry/affine2d.h>
#include <inspirecv/task/core/matrix.h>

#include <cstring>
#include <cmath>

#if defined(INSPIRECV_TASK_USE_NEON)
#include <arm_neon.h>
#endif

namespace inspirecv {
namespace task {
namespace {

geometry::Affine2D AffinePart(const Matrix& matrix) {
    return {matrix.getScaleX(), matrix.getSkewX(), matrix.getTranslateX(),
            matrix.getSkewY(), matrix.getScaleY(), matrix.getTranslateY()};
}

void CopyPointRange(Point* destination, const Point* source, int count) {
    if (destination != source && count > 0) {
        std::memcpy(destination, source, static_cast<size_t>(count) * sizeof(Point));
    }
}

void ApplyOffset(Point* destination, const Point* source, int count,
                 float offsetX, float offsetY) {
#if defined(INSPIRECV_TASK_USE_NEON)
    const float32x4_t offsetX4 = vdupq_n_f32(offsetX);
    const float32x4_t offsetY4 = vdupq_n_f32(offsetY);
    while (count >= 4) {
        float32x4x2_t coordinates = vld2q_f32(reinterpret_cast<const float*>(source));
        coordinates.val[0] = vaddq_f32(coordinates.val[0], offsetX4);
        coordinates.val[1] = vaddq_f32(coordinates.val[1], offsetY4);
        vst2q_f32(reinterpret_cast<float*>(destination), coordinates);
        source += 4;
        destination += 4;
        count -= 4;
    }
#endif
    while (count-- > 0) {
        destination->fX = source->fX + offsetX;
        destination->fY = source->fY + offsetY;
        ++source;
        ++destination;
    }
}

void ApplyAxisAligned(const geometry::Affine2D& transform, Point* destination,
                      const Point* source, int count) {
#if defined(INSPIRECV_TASK_USE_NEON)
    const float32x4_t offsetX4 = vdupq_n_f32(transform.x_offset);
    const float32x4_t offsetY4 = vdupq_n_f32(transform.y_offset);
    const float32x4_t scaleX4 = vdupq_n_f32(transform.xx);
    const float32x4_t scaleY4 = vdupq_n_f32(transform.yy);
    while (count >= 4) {
        float32x4x2_t coordinates = vld2q_f32(reinterpret_cast<const float*>(source));
        coordinates.val[0] = vaddq_f32(vmulq_f32(coordinates.val[0], scaleX4), offsetX4);
        coordinates.val[1] = vaddq_f32(vmulq_f32(coordinates.val[1], scaleY4), offsetY4);
        vst2q_f32(reinterpret_cast<float*>(destination), coordinates);
        source += 4;
        destination += 4;
        count -= 4;
    }
#endif
    while (count-- > 0) {
        destination->fX = source->fX * transform.xx + transform.x_offset;
        destination->fY = source->fY * transform.yy + transform.y_offset;
        ++source;
        ++destination;
    }
}

void ApplyAffine(const geometry::Affine2D& transform, Point* destination,
                 const Point* source, int count) {
#if defined(INSPIRECV_TASK_USE_NEON)
    const float32x4_t offsetX4 = vdupq_n_f32(transform.x_offset);
    const float32x4_t offsetY4 = vdupq_n_f32(transform.y_offset);
    const float32x4_t xx4 = vdupq_n_f32(transform.xx);
    const float32x4_t xy4 = vdupq_n_f32(transform.xy);
    const float32x4_t yx4 = vdupq_n_f32(transform.yx);
    const float32x4_t yy4 = vdupq_n_f32(transform.yy);
    // Preserve the original implementation's two execution shapes. Batches
    // larger than four points use deinterleaved vectors; the final two-point
    // groups use one interleaved vector. Apart from avoiding a shuffle in the
    // wide loop, this fixes the floating-point accumulation order consumed by
    // the scanline sampler, so it is part of the observable numeric contract.
    if (count > 4) {
        while (count >= 4) {
            const float32x4x2_t input = vld2q_f32(reinterpret_cast<const float*>(source));
            float32x4x2_t output;
            output.val[0] = offsetX4 + input.val[0] * xx4 + input.val[1] * xy4;
            output.val[1] = offsetY4 + input.val[0] * yx4 + input.val[1] * yy4;
            vst2q_f32(reinterpret_cast<float*>(destination), output);
            source += 4;
            destination += 4;
            count -= 4;
        }
    }

    if ((count & 1) != 0) {
        const float x = source->fX;
        const float y = source->fY;
#if defined(__clang__) && defined(__aarch64__) && !INSPIRECV_TASK_PRESERVE_LTO_EVALUATION
        // Preserve the historical odd-tail FMA order. In particular, its Y
        // expression is ordered differently from the interleaved pair loop.
        destination->fX = std::fma(y, transform.xy,
                                   std::fma(x, transform.xx, transform.x_offset));
        destination->fY = std::fma(y, transform.yy,
                                   std::fma(x, transform.yx, transform.y_offset));
#else
        destination->fX = x * transform.xx + y * transform.xy + transform.x_offset;
        destination->fY = x * transform.yx + y * transform.yy + transform.y_offset;
#endif
        ++source;
        ++destination;
        --count;
    }

    const float pairOffsets[] = {transform.x_offset, transform.y_offset,
                                 transform.x_offset, transform.y_offset};
    const float pairScales[] = {transform.xx, transform.yy,
                                transform.xx, transform.yy};
    const float pairSkews[] = {transform.xy, transform.yx,
                               transform.xy, transform.yx};
    const float32x4_t offsetPairs = vld1q_f32(pairOffsets);
    const float32x4_t scalePairs = vld1q_f32(pairScales);
    const float32x4_t skewPairs = vld1q_f32(pairSkews);
    while (count >= 2) {
        const float32x4_t input = vld1q_f32(reinterpret_cast<const float*>(source));
        const float32x4_t swapped = vrev64q_f32(input);
        float32x4_t output = vmlaq_f32(offsetPairs, input, scalePairs);
        output = vmlaq_f32(output, swapped, skewPairs);
        vst1q_f32(reinterpret_cast<float*>(destination), output);
        source += 2;
        destination += 2;
        count -= 2;
    }
#else
    while (count-- > 0) {
        const float x = source->fX;
        const float y = source->fY;
        destination->fX = x * transform.xx + y * transform.xy + transform.x_offset;
        destination->fY = x * transform.yx + y * transform.yy + transform.y_offset;
        ++source;
        ++destination;
    }
#endif
}

void ApplyProjective(const Matrix& transform, Point* destination,
                     const Point* source, int count) {
    while (count-- > 0) {
        const float x = source->fX;
        const float y = source->fY;
        const float projectedX = x * transform.getScaleX() +
                                 y * transform.getSkewX() + transform.getTranslateX();
        const float projectedY = x * transform.getSkewY() +
                                 y * transform.getScaleY() + transform.getTranslateY();
        float divisor = x * transform.getPerspX() +
                        (y * transform.getPerspY() + transform[Matrix::kMPersp2]);
        if (divisor != 0.0f) {
            divisor = 1.0f / divisor;
        }
        destination->fX = projectedX * divisor;
        destination->fY = projectedY * divisor;
        ++source;
        ++destination;
    }
}

void StoreOrderedBounds(float firstX, float firstY, float secondX, float secondY,
                        Rect* destination) {
    destination->fLeft = secondX < firstX ? secondX : firstX;
    destination->fTop = secondY < firstY ? secondY : firstY;
    destination->fRight = firstX > secondX ? firstX : secondX;
    destination->fBottom = firstY > secondY ? firstY : secondY;
}

}  // namespace

void Matrix::mapPointsGeneral(Point destination[], const Point source[], int count) const {
    const TypeMask type = getType();
    if (type == kIdentity_Mask) {
        CopyPointRange(destination, source, count);
    } else if (type == kTranslate_Mask) {
        ApplyOffset(destination, source, count, getTranslateX(), getTranslateY());
    } else if ((type & (kAffine_Mask | kPerspective_Mask)) == 0) {
        ApplyAxisAligned(AffinePart(*this), destination, source, count);
    } else if ((type & kPerspective_Mask) == 0) {
        ApplyAffine(AffinePart(*this), destination, source, count);
    } else {
        ApplyProjective(*this, destination, source, count);
    }
}

void Matrix::mapXYGeneral(float x, float y, Point* result) const {
    const TypeMask type = getType();
    if (type == kIdentity_Mask) {
        result->fX = x;
        result->fY = y;
        return;
    }
    if ((type & kPerspective_Mask) != 0) {
        const float projectedX = x * getScaleX() + y * getSkewX() + getTranslateX();
        const float projectedY = x * getSkewY() + y * getScaleY() + getTranslateY();
        float divisor = x * getPerspX() + y * getPerspY() + (*this)[kMPersp2];
        if (divisor != 0.0f) {
            divisor = 1.0f / divisor;
        }
        result->fX = projectedX * divisor;
        result->fY = projectedY * divisor;
        return;
    }
    if ((type & kAffine_Mask) != 0) {
        geometry::MapPointNested(AffinePart(*this), x, y, &result->fX, &result->fY);
        return;
    }
    result->fX = x * getScaleX() + getTranslateX();
    result->fY = y * getScaleY() + getTranslateY();
}

void Matrix::mapRectScaleTranslate(Rect* destination, const Rect& source) const {
    INSPIRECV_TASK_ASSERT(isScaleTranslate());
    const float firstX = source.fLeft * getScaleX() + getTranslateX();
    const float firstY = source.fTop * getScaleY() + getTranslateY();
    const float secondX = source.fRight * getScaleX() + getTranslateX();
    const float secondY = source.fBottom * getScaleY() + getTranslateY();
    StoreOrderedBounds(firstX, firstY, secondX, secondY, destination);
}

bool Matrix::mapRect(Rect* destination, const Rect& source) const {
    if (!isScaleTranslate()) {
        return false;
    }
    mapRectScaleTranslate(destination, source);
    return true;
}

}  // namespace task
}  // namespace inspirecv
