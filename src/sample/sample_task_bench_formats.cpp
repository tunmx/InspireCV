#include <inspirecv/task/task.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>

#if !defined(INSPIRECV_BACKEND_OPENCV) && !defined(INSPIRECV_BACKEND_OKCV_USE_OPENCV)
#error "This sample requires OpenCV enabled (INSPIRECV_BACKEND_OPENCV or INSPIRECV_BACKEND_OKCV_USE_OPENCV)"
#endif

using namespace inspirecv::task;

static cv::Mat toNV21(const cv::Mat& bgr) {
    cv::Mat yuv;
#ifdef CV_COLOR_BGR2YUV_NV21
    cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV_NV21);
    return yuv;
#else
    cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV_I420);
    int H = bgr.rows, W = bgr.cols;
    const uint8_t* src = yuv.data;
    cv::Mat out(H + H/2, W, CV_8UC1);
    std::memcpy(out.data, src, W*H);
    const uint8_t* U = src + W*H;
    const uint8_t* V = U + (W/2)*(H/2);
    uint8_t* VU = out.data + W*H;
    for (int r=0;r<H/2;++r){ for (int c=0;c<W/2;++c){ VU[r*W + 2*c + 0] = V[r*(W/2)+c]; VU[r*W + 2*c + 1] = U[r*(W/2)+c]; }}
    return out;
#endif
}

static cv::Mat toNV12(const cv::Mat& bgr) {
    cv::Mat yuv;
#ifdef CV_COLOR_BGR2YUV_NV12
    cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV_NV12);
    return yuv;
#else
    cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV_I420);
    int H = bgr.rows, W = bgr.cols;
    const uint8_t* src = yuv.data;
    cv::Mat out(H + H/2, W, CV_8UC1);
    std::memcpy(out.data, src, W*H);
    const uint8_t* U = src + W*H;
    const uint8_t* V = U + (W/2)*(H/2);
    uint8_t* UV = out.data + W*H;
    for (int r=0;r<H/2;++r){ for (int c=0;c<W/2;++c){ UV[r*W + 2*c + 0] = U[r*(W/2)+c]; UV[r*W + 2*c + 1] = V[r*(W/2)+c]; }}
    return out;
#endif
}

static cv::Mat toI420(const cv::Mat& bgr) { cv::Mat yuv; cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV_I420); return yuv; }

struct Case { const char* name; StreamFormat srcFmt; const uint8_t* src; int iw, ih; int stride; };

struct AlignedBuf {
    uint8_t* ptr{nullptr}; size_t size{0};
    void alloc(size_t n, size_t align=64) {
        free(); size = n; void* p=nullptr; posix_memalign(&p, align, n); ptr = (uint8_t*)p;
    }
    void free(){ if(ptr){ ::free(ptr); ptr=nullptr; size=0; }}
    ~AlignedBuf(){ free(); }
};

static double benchCase(const Case& c, int loops) {
    StreamTask::Config cfg; cfg.sourceFormat=c.srcFmt; cfg.destFormat=BGR; cfg.filterType=NEAREST; cfg.wrap=CLAMP_TO_EDGE;
    auto proc = StreamTask::create(cfg);
    Matrix m; m.setIdentity(); proc->setMatrix(m);
    AlignedBuf dst; dst.alloc((size_t)c.iw * c.ih * 3);
    proc->convert(c.src, c.iw, c.ih, c.stride, dst.ptr, c.iw, c.ih, 3, 0, halide_type_of<uint8_t>());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i=0;i<loops;++i) {
        proc->convert(c.src, c.iw, c.ih, c.stride, dst.ptr, c.iw, c.ih, 3, 0, halide_type_of<uint8_t>());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    StreamTask::destroy(proc);
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / loops;
}

static double benchCaseAffine(const Case& c, int loops, const cv::Mat& M) {
    StreamTask::Config cfg; cfg.sourceFormat=c.srcFmt; cfg.destFormat=BGR; cfg.filterType=BILINEAR; cfg.wrap=CLAMP_TO_EDGE;
    auto proc = StreamTask::create(cfg);
    cv::Mat Minv; cv::invertAffineTransform(M, Minv);
    auto getf = [&](int r, int col)->float { return Minv.depth()==CV_64F ? (float)Minv.at<double>(r,col) : Minv.at<float>(r,col); };
    Matrix mm; mm.setAll(getf(0,0), getf(0,1), getf(0,2), getf(1,0), getf(1,1), getf(1,2), 0, 0, 1);
    proc->setMatrix(mm);
    AlignedBuf dst; dst.alloc((size_t)c.iw * c.ih * 3);
    proc->convert(c.src, c.iw, c.ih, c.stride, dst.ptr, c.iw, c.ih, 3, 0, halide_type_of<uint8_t>());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i=0;i<loops;++i) {
        proc->convert(c.src, c.iw, c.ih, c.stride, dst.ptr, c.iw, c.ih, 3, 0, halide_type_of<uint8_t>());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    StreamTask::destroy(proc);
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / loops;
}

int main(int argc, char** argv) {
    const char* path = "../woman.png";
    if (argc > 1) path = argv[1];
    int loops = 1000;
    if (argc > 2) loops = std::max(1, atoi(argv[2]));

    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) { std::printf("Failed to read %s\n", path); return 1; }
    if (bgr.cols % 2 || bgr.rows % 2) bgr = bgr(cv::Rect(0,0,bgr.cols & ~1, bgr.rows & ~1)).clone();
    int W=bgr.cols, H=bgr.rows;

    cv::Mat rgb, bgra, gray;
    cv::cvtColor(bgr, rgb,  cv::COLOR_BGR2RGB);
    cv::cvtColor(bgr, bgra, cv::COLOR_BGR2BGRA);
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    auto nv21 = toNV21(bgr);
    auto nv12 = toNV12(bgr);
    auto i420 = toI420(bgr);

    std::vector<Case> cases = {
        {"BGR->BGR",   BGR,   bgr.data,  W, H, W*3},
        {"RGB->BGR",   RGB,   rgb.data,  W, H, W*3},
        {"BGRA->BGR",  BGRA,  bgra.data, W, H, W*4},
        {"RGBA->BGR",  RGBA,  nullptr,   0, 0, 0},
        {"GRAY->BGR",  GRAY,  gray.data, W, H, W*1},
        {"NV21->BGR",  YUV_NV21, nv21.data, W, H, 0},
        {"NV12->BGR",  YUV_NV12, nv12.data, W, H, 0},
        {"I420->BGR",  YUV_I420, i420.data, W, H, 0},
    };

    std::printf("loops=%d\n", loops);
    for (const auto& c : cases) {
        if (c.src == nullptr) continue;
        double avg = benchCase(c, loops);
        std::printf("%-10s  %8.3f ms\n", c.name, avg);
    }

    std::printf("\n-- With affine (BILINEAR) --\n");
    cv::Point2f center(W*0.5f, H*0.5f);
    std::vector<std::pair<const char*, cv::Mat>> mats;
    mats.push_back({"identity", cv::Mat::eye(2,3,CV_32F)});
    mats.push_back({"translate", (cv::Mat_<float>(2,3) << 1,0,20, 0,1,-15)});
    mats.push_back({"scale0.5", (cv::Mat_<float>(2,3) << 0.5f,0,0, 0,0.5f,0)});
    mats.push_back({"scale2.0", (cv::Mat_<float>(2,3) << 2.0f,0,0, 0,2.0f,0)});
    mats.push_back({"rot30", cv::getRotationMatrix2D(center, 30.0, 1.0)});
    mats.push_back({"rot-45", cv::getRotationMatrix2D(center, -45.0, 1.0)});

    for (const auto& kv : mats) {
        const char* mname = kv.first; const cv::Mat& M = kv.second;
        for (const auto& c : cases) {
            if (c.src == nullptr) continue;
            double avg = benchCaseAffine(c, loops, M);
            std::printf("%-6s + %-10s  %8.3f ms\n", mname, c.name, avg);
        }
    }

    return 0;
}


