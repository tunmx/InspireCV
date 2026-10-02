#include "../../common/common.h"
#include "inspirecv/core/runtime/cpu_features.h"
#include <inspirecv/inspirecv.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if !INSPIRECV_TEST_BACKEND_OPENCV
#include "inspirecv/backends/okcv/bitmap/bitmap.h"
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include "inspirecv/backends/okcv/kernels/x86/image_affine_avx2.h"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
#endif

namespace {
using AffineMatrix = std::array<float, 6>;
float AffineAdd(float a, float b) { volatile float result = a + b; return result; }
float AffineMultiply(float a, float b) { volatile float result = a * b; return result; }

std::vector<uint8_t> AffineOracle(const uint8_t* input, int sw, int sh,
                                  int width, int height, int channels,
                                  const AffineMatrix& matrix, bool replicate,
                                  uint8_t border) {
    std::vector<uint8_t> expected(size_t(width) * height * channels);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        float sx = AffineAdd(AffineAdd(AffineMultiply(float(x), matrix[0]),
                                       AffineMultiply(float(y), matrix[1])), matrix[2]);
        float sy = AffineAdd(AffineAdd(AffineMultiply(float(x), matrix[3]),
                                       AffineMultiply(float(y), matrix[4])), matrix[5]);
        const size_t pixel = (size_t(y) * width + x) * channels;
        if (sx < 0 || sy < 0 || sx >= sw || sy >= sh) {
            if (!replicate) {
                std::fill_n(expected.data() + pixel, channels, border);
                continue;
            }
            sx = std::max(0.f, std::min(float(sw - 1), sx));
            sy = std::max(0.f, std::min(float(sh - 1), sy));
        }
        const int x0 = std::min(int(sx), sw - 1), x1 = std::min(x0 + 1, sw - 1);
        const int y0 = std::min(int(sy), sh - 1), y1 = std::min(y0 + 1, sh - 1);
        const float fx = sx - x0, fy = sy - y0;
        for (int c = 0; c < channels; ++c) {
            const auto sample = [&](int ix, int iy) {return float(input[(size_t(iy) * sw + ix) * channels + c]);};
            const float tl = sample(x0,y0), tr = sample(x1,y0), bl = sample(x0,y1), br = sample(x1,y1);
            const float top = AffineAdd(tl, AffineMultiply(tr-tl, fx));
            const float bottom = AffineAdd(bl, AffineMultiply(br-bl, fx));
            const float value = AffineAdd(top, AffineMultiply(bottom-top, fy));
            expected[pixel+c] = uint8_t(std::round(value));
        }
    }
    return expected;
}

void CheckAffineBytes(const uint8_t* actual, const uint8_t* expected, size_t count) {
    size_t mismatch = 0;
    while (mismatch < count && actual[mismatch] == expected[mismatch]) ++mismatch;
    const int got = mismatch < count ? actual[mismatch] : -1;
    const int wanted = mismatch < count ? expected[mismatch] : -1;
    CAPTURE(count, mismatch, got, wanted);
    REQUIRE(mismatch == count);
}

std::vector<AffineMatrix> GeneralMatrices() {
    return {{{1.f,.25f,-.5f, -.125f,1.f,.25f}},
             {{.75f,-.5f,3.25f, .25f,.875f,-1.25f}},
             {{-.5f,.25f,17.f, -.25f,-.5f,9.f}},
             {{0.f,1.f,.5f, -1.f,0.f,17.f}},
             {{.1f,.731f,-3.17f, -.293f,1.127f,5.91f}},
             {{1.0001f,-.0033f,-.4999f, .0001f,.913f,.5001f}},
             {{-1.77f,.027f,39.25f, .381f,-.18f,11.1f}},
             {{1e-5f,.5f,-1e-7f, .5f,-1e-5f,0.f}},
             {{1e30f,1e30f,-1e20f, -1e30f,1e20f,1e30f}}};
}

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
class AffineGuardBuffer {
public:
    explicit AffineGuardBuffer(size_t count) {
#if defined(_WIN32)
        SYSTEM_INFO info; GetSystemInfo(&info); const size_t page = info.dwPageSize;
#else
        const size_t page = size_t(sysconf(_SC_PAGESIZE));
#endif
        const size_t usable = ((count + page - 1) / page) * page;
        size_ = usable + page;
#if defined(_WIN32)
        allocation_ = static_cast<uint8_t*>(VirtualAlloc(nullptr, size_, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE));
        REQUIRE(allocation_ != nullptr);
        DWORD previous = 0;
        REQUIRE(VirtualProtect(allocation_+usable,page,PAGE_NOACCESS,&previous) != 0);
#else
        void* mapping = mmap(nullptr,size_,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        REQUIRE(mapping != MAP_FAILED); allocation_ = static_cast<uint8_t*>(mapping);
        REQUIRE(mprotect(allocation_+usable,page,PROT_NONE) == 0);
#endif
        data_ = allocation_+usable-count;
    }
    ~AffineGuardBuffer() {
#if defined(_WIN32)
        VirtualFree(allocation_,0,MEM_RELEASE);
#else
        munmap(allocation_,size_);
#endif
    }
    uint8_t* data() {return data_;}
    AffineGuardBuffer(const AffineGuardBuffer&) = delete;
    AffineGuardBuffer& operator=(const AffineGuardBuffer&) = delete;
private:uint8_t* allocation_=nullptr;uint8_t* data_=nullptr;size_t size_=0;
};
#endif
} // namespace

TEST_CASE("image_general_affine_c3_preserves_reference_coordinate_and_border_contract",
          "[image][simd-dispatch][affine-general][regression]") {
    const std::vector<AffineMatrix> matrices = {
        {{1.f,.25f,-.5f, -.125f,1.f,.25f}},
        {{.75f,-.5f,3.25f, .25f,.875f,-1.25f}},
        {{-.5f,.25f,17.f, -.25f,-.5f,9.f}},
        {{0.f,1.f,.5f, -1.f,0.f,17.f}}};
    for(uint32_t mask : {uint32_t(0),inspirecv::cpu::kAllCpuFeatures}) {
        const inspirecv::cpu::ScopedCpuFeatureMask features(mask);
        for (int sw : {1,2,3,4,7,8,9,17,33}) for (int sh : {1,3,9}) {
            std::vector<uint8_t> storage(3+size_t(sw)*sh*3+19,0xA7);
            for(size_t i=0;i<size_t(sw)*sh*3;++i)storage[3+i]=uint8_t(i*71+i/7*13);
            const auto before=storage;
            const auto input=inspirecv::Image::Create(sw,sh,3,storage.data()+3,false);
            for(int width : {1,7,8,9,15,16,17,33,65}) for(const auto& m : matrices) {
                constexpr int height=7;
                const auto expected=AffineOracle(input.Data(),sw,sh,width,height,3,m,false,0);
                const auto matrix=inspirecv::TransformMatrix::Create(m[0],m[1],m[2],m[3],m[4],m[5]);
                const auto actual=input.WarpAffine(matrix,width,height);
                CAPTURE(mask,sw,sh,width,m[0],m[1],m[2],m[3],m[4],m[5]);
                CheckAffineBytes(actual.Data(),expected.data(),expected.size());
                REQUIRE(actual.Data()!=input.Data());
            }
            REQUIRE(storage==before);
        }
    }
}

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
TEST_CASE("image_general_affine_avx2_matches_independent_scalar_all_channels",
          "[image][simd-dispatch][affine-general][kernel]") {
    if(!inspirecv::cpu::HasAvx2()) {SUCCEED("AVX2 unavailable: optional kernel not executed");return;}
    for(int channels : {1,3,4}) for(int sw : {1,2,3,4,7,8,9,17,33}) for(int sh : {1,3,9}) {
        const size_t count=size_t(sw)*sh*channels;
        for(size_t offset : {size_t(0),size_t(1),size_t(7),size_t(15)}) {
            std::vector<uint8_t> source(offset+count+19,0xA7);
            for(size_t i=0;i<count;++i)source[offset+i]=uint8_t(i*73+i/13*17);
            const auto original=source;
            for(int width : {1,7,8,9,15,16,17,33,65}) for(const auto& m : GeneralMatrices()) for(bool replicate : {false,true}) {
                constexpr int height=7;
                const uint8_t border=replicate?0:173;
                const auto reference=AffineOracle(source.data()+offset,sw,sh,width,height,channels,m,replicate,border);
                std::vector<uint8_t> output(3+reference.size()+19,0xCD),expected=output;
                std::copy(reference.begin(),reference.end(),expected.begin()+3);
                CAPTURE(channels,sw,sh,offset,width,replicate,m[0],m[1],m[2],m[3],m[4],m[5]);
                REQUIRE(okcv::x86::AffineBilinearReferenceU8Avx2(source.data()+offset,sw,sh,output.data()+3,width,height,channels,m.data(),replicate,border));
                CheckAffineBytes(output.data(),expected.data(),output.size());
            }
            REQUIRE(source==original);
        }
    }
}

TEST_CASE("image_general_affine_avx2_does_not_read_or_write_beyond_last_row",
          "[image][simd-dispatch][affine-general][guard-page]") {
    if(!inspirecv::cpu::HasAvx2()) {SUCCEED("AVX2 unavailable: optional kernel not executed");return;}
    for(int channels : {1,3,4}) for(int sw : {1,2,3,4,5,7,8,9,17}) for(int width : {7,8,9,15,16,17,33}) {
        constexpr int sh=3,height=5;
        AffineGuardBuffer source(size_t(sw)*sh*channels),output(size_t(width)*height*channels);
        for(size_t i=0;i<size_t(sw)*sh*channels;++i)source.data()[i]=uint8_t(37*i+i/7*19);
        const AffineMatrix m={{.001f,.125f,float(sw)-.75f,0.f,.001f,float(sh)-.25f}};
        for(bool replicate : {false,true}) {
            const auto expected=AffineOracle(source.data(),sw,sh,width,height,channels,m,replicate,173);
            CAPTURE(channels,sw,width,replicate);
            REQUIRE(okcv::x86::AffineBilinearReferenceU8Avx2(source.data(),sw,sh,output.data(),width,height,channels,m.data(),replicate,173));
            CheckAffineBytes(output.data(),expected.data(),expected.size());
        }
    }
}

TEST_CASE("image_general_affine_avx2_rejects_unsafe_transforms_without_writes",
          "[image][simd-dispatch][affine-general][validation]") {
    if(!inspirecv::cpu::HasAvx2()) {SUCCEED("AVX2 unavailable: optional kernel not executed");return;}
    const uint8_t source[12]={};
    const float invalid[]={std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()};
    for(int element=0;element<6;++element)for(float value:invalid) {
        AffineMatrix m={{1,.25f,0,-.25f,1,0}};m[element]=value;
        std::vector<uint8_t> output(128,0xCD),original=output;
        REQUIRE_FALSE(okcv::x86::AffineBilinearReferenceU8Avx2(source,2,2,output.data(),8,4,3,m.data(),false,0));
        REQUIRE(output==original);
    }
    const AffineMatrix overflow={{std::numeric_limits<float>::max(),.25f,0,-.25f,1,0}};
    std::vector<uint8_t> output(128,0xCD),original=output;
    REQUIRE_FALSE(okcv::x86::AffineBilinearReferenceU8Avx2(source,2,2,output.data(),8,4,3,overflow.data(),false,0));
    REQUIRE(output==original);
    const AffineMatrix normal={{1,.25f,0,-.25f,1,0}};
    REQUIRE_FALSE(okcv::x86::AffineBilinearReferenceU8Avx2(source,std::numeric_limits<int>::max(),2,output.data(),8,4,3,normal.data(),false,0));
    REQUIRE(output==original);
}

TEST_CASE("image_general_affine_avx2_rounds_true_half_without_double_rounding",
          "[image][simd-dispatch][affine-general][rounding]") {
    if(!inspirecv::cpu::HasAvx2()) {SUCCEED("AVX2 unavailable: optional kernel not executed");return;}
    for(int channels:{1,3,4})for(float x:{std::nextafter(.5f,0.f),.5f,std::nextafter(.5f,1.f)}) {
        std::vector<uint8_t> source(channels*2,0);
        std::fill(source.begin()+channels,source.end(),1);
        const AffineMatrix m={{0,.25f,x,0,0,0}};
        std::vector<uint8_t> actual(33*channels,0xCD),expected(actual.size(),x<.5f?0:1);
        REQUIRE(okcv::x86::AffineBilinearReferenceU8Avx2(source.data(),2,1,actual.data(),33,1,channels,m.data(),false,0));
        CheckAffineBytes(actual.data(),expected.data(),actual.size());
    }
}
#endif
#endif
