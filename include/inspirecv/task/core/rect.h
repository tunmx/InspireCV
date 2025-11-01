#ifndef INSPIRECV_STREAMTASK_CORE_RECT_H_
#define INSPIRECV_STREAMTASK_CORE_RECT_H_

#include <math.h>
#include <algorithm>
#include <utility>
#include "st_defs.h"

namespace inspirecv {
namespace task {

struct Point {
    float fX;
    float fY;

    void set(float x, float y) {
        fX = x;
        fY = y;
    }
};

struct INSPIRECV_TASK_PUBLIC Rect {
    float fLeft;   //!< smaller x-axis bounds
    float fTop;    //!< smaller y-axis bounds
    float fRight;  //!< larger x-axis bounds
    float fBottom; //!< larger y-axis bounds

    static constexpr Rect MakeEmpty() {
        return Rect{0, 0, 0, 0};
    }

    static constexpr Rect MakeWH(float w, float h) {
        return Rect{0, 0, w, h};
    }

    static Rect MakeIWH(int w, int h) {
        Rect r;
        r.set(0, 0, (float)(w), (float)(h));
        return r;
    }

    static constexpr Rect MakeLTRB(float l, float t, float r, float b) {
        return Rect{l, t, r, b};
    }

    static constexpr Rect MakeXYWH(float x, float y, float w, float h) {
        return Rect{x, y, x + w, y + h};
    }

    bool isEmpty() const {
        return !(fLeft < fRight && fTop < fBottom);
    }

    bool isSorted() const {
        return fLeft <= fRight && fTop <= fBottom;
    }

    float x() const { return fLeft; }
    float y() const { return fTop; }
    float left() const { return fLeft; }
    float top() const { return fTop; }
    float right() const { return fRight; }
    float bottom() const { return fBottom; }

    float width() const { return fRight - fLeft; }
    float height() const { return fBottom - fTop; }

    float centerX() const {
        return 0.5f * (fLeft) + 0.5f * (fRight);
    }
    float centerY() const {
        return 0.5f * (fTop) + 0.5f * (fBottom);
    }

    void setEmpty() { *this = MakeEmpty(); }

    void set(float left, float top, float right, float bottom) {
        fLeft   = left;
        fTop    = top;
        fRight  = right;
        fBottom = bottom;
    }
    void setLTRB(float left, float top, float right, float bottom) {
        this->set(left, top, right, bottom);
    }
    void iset(int left, int top, int right, int bottom) {
        fLeft   = (float)(left);
        fTop    = (float)(top);
        fRight  = (float)(right);
        fBottom = (float)(bottom);
    }
    void isetWH(int width, int height) {
        fLeft = fTop = 0;
        fRight       = (float)(width);
        fBottom      = (float)(height);
    }
    void setXYWH(float x, float y, float width, float height) {
        fLeft   = x;
        fTop    = y;
        fRight  = x + width;
        fBottom = y + height;
    }
    void setWH(float width, float height) {
        fLeft   = 0;
        fTop    = 0;
        fRight  = width;
        fBottom = height;
    }

    Rect makeOffset(float dx, float dy) const {
        return MakeLTRB(fLeft + dx, fTop + dy, fRight + dx, fBottom + dy);
    }
    Rect makeInset(float dx, float dy) const {
        return MakeLTRB(fLeft + dx, fTop + dy, fRight - dx, fBottom - dy);
    }
    Rect makeOutset(float dx, float dy) const {
        return MakeLTRB(fLeft - dx, fTop - dy, fRight + dx, fBottom + dy);
    }

    void offset(float dx, float dy) {
        fLeft += dx;
        fTop += dy;
        fRight += dx;
        fBottom += dy;
    }
    void offsetTo(float newX, float newY) {
        fRight += newX - fLeft;
        fBottom += newY - fTop;
        fLeft = newX;
        fTop  = newY;
    }
    void inset(float dx, float dy) {
        fLeft += dx;
        fTop += dy;
        fRight -= dx;
        fBottom -= dy;
    }
    void outset(float dx, float dy) {
        this->inset(-dx, -dy);
    }

private:
    static bool Intersects(float al, float at, float ar, float ab, float bl, float bt, float br, float bb) {
        float L = std::max(al, bl);
        float R = std::min(ar, br);
        float T = std::max(at, bt);
        float B = std::min(ab, bb);
        return L < R && T < B;
    }

public:
    bool intersects(float left, float top, float right, float bottom) const {
        return Intersects(fLeft, fTop, fRight, fBottom, left, top, right, bottom);
    }
    bool intersects(const Rect& r) const {
        return Intersects(fLeft, fTop, fRight, fBottom, r.fLeft, r.fTop, r.fRight, r.fBottom);
    }
    static bool Intersects(const Rect& a, const Rect& b) {
        return Intersects(a.fLeft, a.fTop, a.fRight, a.fBottom, b.fLeft, b.fTop, b.fRight, b.fBottom);
    }

    void joinNonEmptyArg(const Rect& r) {
        INSPIRECV_TASK_ASSERT(!r.isEmpty());
        if (fLeft >= fRight || fTop >= fBottom) {
            *this = r;
        } else {
            this->joinPossiblyEmptyRect(r);
        }
    }
    void joinPossiblyEmptyRect(const Rect& r) {
        fLeft   = std::min(fLeft, r.left());
        fTop    = std::min(fTop, r.top());
        fRight  = std::max(fRight, r.right());
        fBottom = std::max(fBottom, r.bottom());
    }

    bool contains(float x, float y) const {
        return x >= fLeft && x < fRight && y >= fTop && y < fBottom;
    }

    void sort() {
        using std::swap;
        if (fLeft > fRight) {
            swap(fLeft, fRight);
        }
        if (fTop > fBottom) {
            swap(fTop, fBottom);
        }
    }
    Rect makeSorted() const {
        return MakeLTRB(std::min(fLeft, fRight), std::min(fTop, fBottom), std::max(fLeft, fRight),
                        std::max(fTop, fBottom));
    }

    const float* asScalars() const { return &fLeft; }
};

} // namespace task
} // namespace inspirecv

#endif // INSPIRECV_STREAMTASK_CORE_RECT_H_



