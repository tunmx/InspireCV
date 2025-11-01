#include <inspirecv/task/task.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cmath>

#if !defined(INSPIRECV_BACKEND_OPENCV) && !defined(INSPIRECV_BACKEND_OKCV_USE_OPENCV)
#error "This sample requires OpenCV enabled (INSPIRECV_BACKEND_OPENCV or INSPIRECV_BACKEND_OKCV_USE_OPENCV)"
#endif

using namespace inspirecv::task;

static void multiply3x3(const float a[9], const float b[9], float out[9]) {
    for (int r=0; r<3; ++r) {
        for (int c=0; c<3; ++c) {
            out[r*3+c] = a[r*3+0]*b[0*3+c] + a[r*3+1]*b[1*3+c] + a[r*3+2]*b[2*3+c];
        }
    }
}

static Matrix makeTransform(float cx, float cy, float ow, float oh, float scale, float degrees) {
    float rad = degrees * float(M_PI) / 180.0f;
    float c = std::cos(rad), s = std::sin(rad);
    float T2[9] = {1,0,cx, 0,1,cy, 0,0,1};
    float R[9]  = {c,-s,0,  s,c,0,  0,0,1};
    float S[9]  = {scale,0,0,  0,scale,0,  0,0,1};
    float T1[9] = {1,0,-ow*0.5f,  0,1,-oh*0.5f,  0,0,1};

    float RS[9]; multiply3x3(R, S, RS);
    float RS_T1[9]; multiply3x3(RS, T1, RS_T1);
    float M[9]; multiply3x3(T2, RS_T1, M);

    Matrix mat; mat.setAll(M[0], M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8]);
    return mat;
}

int main(int argc, char** argv) {
    const char* path = "../woman.png";
    if (argc > 1) path = argv[1];

    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR); // BGR
    if (img.empty()) {
        std::printf("Failed to load image: %s\n", path);
        return 1;
    }
    int iw = img.cols, ih = img.rows, ic = 3;
    std::printf("Loaded %dx%d\n", iw, ih);

    StreamTask::Config cfg;
    cfg.sourceFormat = BGR;
    cfg.destFormat = BGRA;
    cfg.filterType = BILINEAR;
    cfg.wrap = CLAMP_TO_EDGE;
    StreamTask* proc = StreamTask::create(cfg);

    int ow = 128, oh = 128, oc = 4;
    std::vector<uint8_t> dst(ow * oh * oc);

    {
        Matrix m;
        m.setAll(0.181011f, 0.0631488f, -32.825f,
                 -0.0631488f, 0.181011f, -4.49382f,
                 0.0f, 0.0f, 1.0f);
        m.invert(&m);
        proc->setMatrix(m);
    }

    proc->convert(img.data, iw, ih, iw * ic, dst.data(), ow, oh, oc, 0, halide_type_of<uint8_t>());

    const int loops = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<loops; ++i) {
        proc->convert(img.data, iw, ih, iw * ic, dst.data(), ow, oh, oc, 0, halide_type_of<uint8_t>());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("loops=%d total=%.3f ms avg=%.3f ms\n", loops, ms, ms/loops);

    cv::Mat outBGRA(oh, ow, CV_8UC4, dst.data());
    cv::Mat outBGR; cv::cvtColor(outBGRA, outBGR, cv::COLOR_BGRA2BGR);
    cv::imwrite("task_bench_out.png", outBGR);

    StreamTask::destroy(proc);
    return 0;
}


