#ifndef INSPIRECV_STREAMTASK_CORE_STREAM_TASK_UTILS_H_
#define INSPIRECV_STREAMTASK_CORE_STREAM_TASK_UTILS_H_

#include <inspirecv/task/core/stream_task.h>
#include <inspirecv/task/core/matrix.h>
#include <inspirecv/task/core/st_types.h>
#include <inspirecv/task/task_status.h>

namespace inspirecv {

using BLITTER = void (*)(const unsigned char* source, unsigned char* dest, size_t count);
using BLIT_FLOAT = void (*)(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
using SAMPLER = void (*)(const unsigned char* source, unsigned char* dest, task::Point* points, size_t sta, size_t count,
                        size_t capacity, size_t iw, size_t ih, size_t yStride);

class StreamTaskUtils {
public:
    struct InsideProperty;
public:
    explicit StreamTaskUtils(const task::StreamTask::Config& config);
    StreamTaskUtils(const StreamTaskUtils&) = delete;
    StreamTaskUtils& operator=(const StreamTaskUtils&) = delete;
    ~StreamTaskUtils();
    // MixedCase API
    static void Destroy(StreamTaskUtils* imageProcessUtils);

    // Compatibility wrappers (will be removed later)
    static inline void destroy(StreamTaskUtils* p) { Destroy(p); }

    inline const task::Matrix& Matrix() const { return transform_; }
    inline const task::Matrix& matrix() const { return Matrix(); }
    void SetMatrix(const task::Matrix& matrix);
    inline void setMatrix(const task::Matrix& m) { SetMatrix(m); }
    void SetPadding(uint8_t value) { padding_value_ = value; }
    inline void setPadding(uint8_t v) { SetPadding(v); }

    task::Matrix transform_;
    task::Matrix transform_invert_;
    InsideProperty* inside_;
    uint8_t padding_value_ = 0;

    BLITTER Choose(task::StreamFormat source, task::StreamFormat dest);
    inline BLITTER choose(task::StreamFormat s, task::StreamFormat d) { return Choose(s, d); }
    BLITTER Choose(int channelByteSize);
    inline BLITTER choose(int bytes) { return Choose(bytes); }
    BLIT_FLOAT Choose(task::StreamFormat format, int dstBpp = 0);
    inline BLIT_FLOAT choose(task::StreamFormat f, int dstBpp = 0) { return Choose(f, dstBpp); }
    SAMPLER Choose(task::StreamFormat format, task::Filter type, bool identity);
    inline SAMPLER choose(task::StreamFormat f, task::Filter t, bool id) { return Choose(f, t, id); }

    void SetDraw();
    inline void setDraw() { SetDraw(); }
    void Draw(uint8_t* img, int w, int h, int c, const int* regions, int num, uint8_t* color);
    inline void draw(uint8_t* img, int w, int h, int c, const int* regions, int num, uint8_t* color) {
        Draw(img, w, h, c, regions, num, color);
    }

    TaskStatus TransformImage(const uint8_t* source, uint8_t* dst, uint8_t* samplerDest, uint8_t* blitDest, int tileCount, int destBytes, const int32_t* regions);
    inline TaskStatus transformImage(const uint8_t* s, uint8_t* d, uint8_t* sd, uint8_t* bd, int tc, int db, const int32_t* r) {
        return TransformImage(s, d, sd, bd, tc, db, r);
    }

    TaskStatus SelectImageProcer(bool identity=true, bool hasBackend=false, bool isDraw = false);
    inline TaskStatus selectImageProcer(bool identity=true, bool hasBackend=false, bool isDraw = false) {
        return SelectImageProcer(identity, hasBackend, isDraw);
    }
    TaskStatus ExecFunc(const uint8_t* source, int stride, void* dest);
    inline TaskStatus execFunc(const uint8_t* s, int stride, void* d) { return ExecFunc(s, stride, d); }
    TaskStatus ResizeFunc(int ic, int iw, int ih, int oc, int ow, int oh, halide_type_t type, int stride = 0);
    inline TaskStatus resizeFunc(int ic, int iw, int ih, int oc, int ow, int oh, halide_type_t t, int stride = 0) {
        return ResizeFunc(ic, iw, ih, oc, ow, oh, t, stride);
    }

private:
};

} // namespace inspirecv

#endif /* INSPIRECV_TASK_STREAMTASKUTILS_HPP */


