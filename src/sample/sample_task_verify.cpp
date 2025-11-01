#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <inspirecv/task/task.h>
#include <inspirecv/core/transform_matrix.h>

using namespace inspirecv::task;

static std::string gOutDir = ".";
static std::string sanitize(const std::string& s) {
    std::string r; r.reserve(s.size());
    for (char c : s) {
        if ((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')) r.push_back(c);
        else r.push_back('_');
    }
    return r;
}

static cv::Mat convertForSave(const cv::Mat& img, StreamFormat fmt) {
    cv::Mat out;
    switch (fmt) {
        case RGB:
            cv::cvtColor(img, out, cv::COLOR_RGB2BGR);
            break;
        case RGBA:
            cv::cvtColor(img, out, cv::COLOR_RGBA2BGR);
            break;
        case BGRA:
            cv::cvtColor(img, out, cv::COLOR_BGRA2BGR);
            break;
        case GRAY:
            cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
            break;
        default:
            out = img; // assume BGR
            break;
    }
    return out;
}

static void saveCompare(const std::string& pathBase, const cv::Mat& cvRef, const cv::Mat& ours, StreamFormat fmt) {
    cv::Mat left = convertForSave(cvRef, fmt);
    cv::Mat right = convertForSave(ours, fmt);
    int h = std::max(left.rows, right.rows);
    int w = left.cols + right.cols;
    cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(0,0,0));
    left.copyTo(canvas(cv::Rect(0, 0, left.cols, left.rows)));
    right.copyTo(canvas(cv::Rect(left.cols, 0, right.cols, right.rows)));
    int baseline = 0;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.6; int thickness = 2;
    cv::putText(canvas, "OpenCV", cv::Point(10, 25), font, scale, cv::Scalar(0,0,0), thickness+2, cv::LINE_AA);
    cv::putText(canvas, "OpenCV", cv::Point(10, 25), font, scale, cv::Scalar(255,255,255), thickness, cv::LINE_AA);
    cv::putText(canvas, "Ours", cv::Point(left.cols + 10, 25), font, scale, cv::Scalar(0,0,0), thickness+2, cv::LINE_AA);
    cv::putText(canvas, "Ours", cv::Point(left.cols + 10, 25), font, scale, cv::Scalar(255,255,255), thickness, cv::LINE_AA);
    cv::imwrite(pathBase + "_cmp.png", canvas);
}

static void saveCompareNamed(const std::string& pathBase, const cv::Mat& left, const char* leftName,
                             const cv::Mat& right, const char* rightName) {
    cv::Mat l = left.channels()==1 ? cv::Mat() : left;
    if (left.channels()==1) cv::cvtColor(left, l, cv::COLOR_GRAY2BGR);
    cv::Mat r = right.channels()==1 ? cv::Mat() : right;
    if (right.channels()==1) cv::cvtColor(right, r, cv::COLOR_GRAY2BGR);
    int h = std::max(l.rows, r.rows);
    int w = l.cols + r.cols;
    cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(0,0,0));
    l.copyTo(canvas(cv::Rect(0, 0, l.cols, l.rows)));
    r.copyTo(canvas(cv::Rect(l.cols, 0, r.cols, r.rows)));
    int font = cv::FONT_HERSHEY_SIMPLEX; double scale=0.6; int thickness=2;
    cv::putText(canvas, leftName, cv::Point(10, 25), font, scale, cv::Scalar(0,0,0), thickness+2, cv::LINE_AA);
    cv::putText(canvas, leftName, cv::Point(10, 25), font, scale, cv::Scalar(255,255,255), thickness, cv::LINE_AA);
    cv::putText(canvas, rightName, cv::Point(l.cols + 10, 25), font, scale, cv::Scalar(0,0,0), thickness+2, cv::LINE_AA);
    cv::putText(canvas, rightName, cv::Point(l.cols + 10, 25), font, scale, cv::Scalar(255,255,255), thickness, cv::LINE_AA);
    cv::imwrite(pathBase + "_cmp.png", canvas);
}

static double computePSNR(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat diff; cv::absdiff(a, b, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    double sse = cv::sum(diff)[0];
    if (sse <= 1e-10) return 100.0;
    double mse = sse / (double)(a.total() * a.channels());
    double psnr = 10.0 * std::log10((255.0 * 255.0) / mse);
    return psnr;
}

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
    for (int r = 0; r < H/2; ++r) {
        for (int c = 0; c < W/2; ++c) {
            VU[r*W + 2*c + 0] = V[r*(W/2) + c];
            VU[r*W + 2*c + 1] = U[r*(W/2) + c];
        }
    }
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
    for (int r = 0; r < H/2; ++r) {
        for (int c = 0; c < W/2; ++c) {
            UV[r*W + 2*c + 0] = U[r*(W/2) + c];
            UV[r*W + 2*c + 1] = V[r*(W/2) + c];
        }
    }
    return out;
#endif
}

static cv::Mat toI420(const cv::Mat& bgr) {
    cv::Mat yuv;
    cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV_I420);
    return yuv;
}

static void runConvert(const std::string& name, const uint8_t* src, int iw, int ih, StreamFormat srcFmt,
                       StreamFormat dstFmt, cv::Mat& cvRef, cv::Mat& out, Filter filter = NEAREST) {
    StreamTask::Config cfg;
    cfg.filterType = filter;
    cfg.sourceFormat = srcFmt;
    cfg.destFormat = dstFmt;
    cfg.wrap = CLAMP_TO_EDGE;
    auto proc = StreamTask::create(cfg);
    inspirecv::TransformMatrix tm = inspirecv::TransformMatrix::Identity();
    proc->setMatrix(tm);
    out.create(cvRef.rows, cvRef.cols, cvRef.type());
    auto err = proc->convert(src, iw, ih, 0, out.data, cvRef.cols, cvRef.rows, 0, 0, halide_type_of<uint8_t>());
    StreamTask::destroy(proc);
    double psnr = computePSNR(cvRef, out);
    int maxDiff = 0;
    for (int i=0;i<cvRef.total()*cvRef.channels();++i) {
        int d = std::abs((int)cvRef.data[i] - (int)out.data[i]);
        if (d > maxDiff) maxDiff = d;
    }
    std::printf("[%-18s] PSNR=%.2f dB  maxDiff=%d\n", name.c_str(), psnr, maxDiff);
    auto base = sanitize(name);
    saveCompare(gOutDir + "/" + base, cvRef, out, dstFmt);
}

static void testGeometry(const cv::Mat& bgr) {
    std::puts("-- Geometry (BILINEAR, CLAMP_TO_EDGE) --");
    std::vector<std::pair<std::string, cv::Mat>> mats;
    int W = bgr.cols, H = bgr.rows;
    cv::Point2f center(W*0.5f, H*0.5f);
    mats.push_back({"identity", cv::Mat::eye(2,3,CV_32F)});
    mats.push_back({"translate", (cv::Mat_<float>(2,3) << 1,0,20, 0,1,-15)});
    mats.push_back({"scale0.5", (cv::Mat_<float>(2,3) << 0.5,0,0, 0,0.5,0)});
    mats.push_back({"scale2.0", (cv::Mat_<float>(2,3) << 2,0,0, 0,2,0)});
    mats.push_back({"rot30", cv::getRotationMatrix2D(center, 30.0, 1.0)});
    mats.push_back({"rot-45", cv::getRotationMatrix2D(center, -45.0, 1.0)});

    for (auto& kv : mats) {
        const auto& name = kv.first;
        const auto& M = kv.second;
        cv::Mat cvOut; cv::warpAffine(bgr, cvOut, M, bgr.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        StreamTask::Config cfg; cfg.filterType=BILINEAR; cfg.sourceFormat=BGR; cfg.destFormat=BGR; cfg.wrap=CLAMP_TO_EDGE;
        auto proc = StreamTask::create(cfg);
        cv::Mat Minv; cv::invertAffineTransform(M, Minv);
        auto getf = [&](int r, int c)->float {
            return Minv.depth()==CV_64F ? (float)Minv.at<double>(r,c) : Minv.at<float>(r,c);
        };
        inspirecv::TransformMatrix tm(getf(0,0), getf(0,1), getf(0,2),
                                      getf(1,0), getf(1,1), getf(1,2));
        proc->setMatrix(tm);
        cv::Mat out(bgr.rows, bgr.cols, bgr.type());
        proc->convert(bgr.data, bgr.cols, bgr.rows, 0, out.data, out.cols, out.rows, 0, 0, halide_type_of<uint8_t>());
        StreamTask::destroy(proc);
        double psnr = computePSNR(cvOut, out);
        int maxDiff = 0; for (int i=0;i<cvOut.total()*cvOut.channels();++i) maxDiff = std::max(maxDiff, std::abs((int)cvOut.data[i]- (int)out.data[i]));
        std::printf("[affine %-10s] PSNR=%.2f  maxDiff=%d\n", name.c_str(), psnr, maxDiff);
        auto base = sanitize(std::string("affine_") + name);
        saveCompare(gOutDir + "/" + base, cvOut, out, BGR);
    }
}

static void testNormStd(const cv::Mat& bgr) {
    std::puts("-- Norm/Std round-trip test --");
    const float meanRGB[3] = {0.485f, 0.456f, 0.406f};
    const float stdRGB[3]  = {0.229f, 0.224f, 0.225f};
    float meanBGR_u8[3] = { meanRGB[2]*255.0f, meanRGB[1]*255.0f, meanRGB[0]*255.0f };
    float normalBGR[3]  = { 1.0f/(255.0f*stdRGB[2]), 1.0f/(255.0f*stdRGB[1]), 1.0f/(255.0f*stdRGB[0]) };

    StreamTask::Config cfg; cfg.filterType=NEAREST; cfg.sourceFormat=BGR; cfg.destFormat=BGR; cfg.wrap=CLAMP_TO_EDGE;
    for (int c=0;c<3;++c){ cfg.mean[c]=meanBGR_u8[c]; cfg.normal[c]=normalBGR[c]; }
    auto proc = StreamTask::create(cfg);
    inspirecv::TransformMatrix tm = inspirecv::TransformMatrix::Identity();
    proc->setMatrix(tm);
    std::vector<float> dst((size_t)bgr.cols*bgr.rows*3);
    proc->convert(bgr.data, bgr.cols, bgr.rows, 0, dst.data(), bgr.cols, bgr.rows, 3, 0, halide_type_of<float>());
    StreamTask::destroy(proc);

    cv::Mat rec(bgr.rows, bgr.cols, CV_8UC3);
    for (int y=0;y<bgr.rows;++y){
        const float* rowF = dst.data() + (size_t)y*bgr.cols*3;
        uchar* rowU = rec.ptr<uchar>(y);
        for (int x=0;x<bgr.cols;++x){
            float fb = rowF[3*x+0]; float fg = rowF[3*x+1]; float fr = rowF[3*x+2];
            float vb = fb * (255.0f*stdRGB[2]) + meanBGR_u8[0];
            float vg = fg * (255.0f*stdRGB[1]) + meanBGR_u8[1];
            float vr = fr * (255.0f*stdRGB[0]) + meanBGR_u8[2];
            rowU[3*x+0] = (uchar)std::min(std::max((int)std::round(vb), 0), 255);
            rowU[3*x+1] = (uchar)std::min(std::max((int)std::round(vg), 0), 255);
            rowU[3*x+2] = (uchar)std::min(std::max((int)std::round(vr), 0), 255);
        }
    }
    double psnr = computePSNR(bgr, rec);
    std::printf("[norm-std roundtrip] PSNR=%.2f dB\n", psnr);
    saveCompareNamed(gOutDir + "/normstd_roundtrip", bgr, "Original", rec, "Recovered");
}

int main(int argc, char** argv) {
    std::string imgPath;
    if (argc >= 2) imgPath = argv[1];
    else imgPath = "../woman.png";
    if (argc >= 3) gOutDir = argv[2];
    cv::Mat bgr = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (bgr.empty()) { std::printf("Failed to read %s\n", imgPath.c_str()); return 1; }
    if (bgr.cols % 2 || bgr.rows % 2) bgr = bgr(cv::Rect(0,0,bgr.cols & ~1, bgr.rows & ~1)).clone();

    testGeometry(bgr);

    std::puts("-- Format conversions --");
    {
        cv::Mat cvRef; cv::cvtColor(bgr, cvRef, cv::COLOR_BGR2RGB);
        cv::Mat out; runConvert("BGR->RGB", bgr.data, bgr.cols, bgr.rows, BGR, RGB, cvRef, out);
    }
    {
        auto nv21 = toNV21(bgr);
        cv::Mat cvRef; cv::cvtColor(nv21, cvRef, cv::COLOR_YUV2BGR_NV21);
        cv::Mat out; runConvert("NV21->BGR", nv21.data, bgr.cols, bgr.rows, YUV_NV21, BGR, cvRef, out);
    }
    {
        auto nv12 = toNV12(bgr);
        cv::Mat cvRef; cv::cvtColor(nv12, cvRef, cv::COLOR_YUV2BGR_NV12);
        cv::Mat out; runConvert("NV12->BGR", nv12.data, bgr.cols, bgr.rows, YUV_NV12, BGR, cvRef, out);
    }
    {
        auto i420 = toI420(bgr);
        cv::Mat cvRef; cv::cvtColor(i420, cvRef, cv::COLOR_YUV2BGR_I420);
        cv::Mat out; runConvert("I420->BGR", i420.data, bgr.cols, bgr.rows, YUV_I420, BGR, cvRef, out);
    }

    testNormStd(bgr);

    return 0;
}




