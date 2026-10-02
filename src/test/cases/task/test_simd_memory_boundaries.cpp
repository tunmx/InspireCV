#include "../../common/common.h"
#include <inspirecv/inspirecv.h>
#include "inspirecv/core/runtime/cpu_features.h"
#include "inspirecv/task/kernels/cpu/channel_ops.h"
#include "inspirecv/task/kernels/cpu/color_ops.h"
#include "inspirecv/task/kernels/cpu/sampling_ops.h"
#include "inspirecv/task/kernels/cpu/tensor_writers.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(_WIN32) || defined(__unix__) || defined(__APPLE__)
namespace {
using namespace inspirecv::task;
using inspirecv::cpu::ScopedCpuFeatureMask;
// Place the final byte immediately before an inaccessible page. Sentinel bytes
// cannot reveal reads past the end; this catches accidental full-vector reads,
// including packed C3 tails and C1 gathers at the rightmost source pixel.
class BoundaryBuffer {
public:
    explicit BoundaryBuffer(size_t size):size_(size) {
#if defined(_WIN32)
        SYSTEM_INFO info;GetSystemInfo(&info);page_=info.dwPageSize;
#else
        page_=size_t(sysconf(_SC_PAGESIZE));
#endif
        const size_t pages=(size+page_-1)/page_;
        mapped_=(pages+2)*page_;
#if defined(_WIN32)
        base_=static_cast<uint8_t*>(VirtualAlloc(nullptr,mapped_,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
        REQUIRE(base_!=nullptr);
        DWORD old=0;
        REQUIRE(VirtualProtect(base_,page_,PAGE_NOACCESS,&old)!=0);
        REQUIRE(VirtualProtect(base_+(pages+1)*page_,page_,PAGE_NOACCESS,&old)!=0);
#else
        void* memory=mmap(nullptr,mapped_,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
        REQUIRE(memory!=MAP_FAILED);base_=static_cast<uint8_t*>(memory);
        REQUIRE(mprotect(base_,page_,PROT_NONE)==0);
        REQUIRE(mprotect(base_+(pages+1)*page_,page_,PROT_NONE)==0);
#endif
        data_=base_+(pages+1)*page_-size;
        std::memset(data_,0xA7,size);
    }
    ~BoundaryBuffer(){
#if defined(_WIN32)
        if(base_)VirtualFree(base_,0,MEM_RELEASE);
#else
        if(base_)munmap(base_,mapped_);
#endif
    }
    uint8_t* data(){return data_;}
    size_t size()const{return size_;}
    BoundaryBuffer(const BoundaryBuffer&)=delete;
    BoundaryBuffer& operator=(const BoundaryBuffer&)=delete;
private:uint8_t* base_=nullptr;uint8_t* data_=nullptr;size_t page_=0,mapped_=0,size_=0;
};
std::vector<uint32_t> BoundaryMasks(){
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    return {0,inspirecv::cpu::kSsse3|inspirecv::cpu::kSse41,inspirecv::cpu::kAllCpuFeatures};
#else
    return {inspirecv::cpu::kAllCpuFeatures};
#endif
}
void Pattern(BoundaryBuffer& buffer){for(size_t i=0;i<buffer.size();++i)buffer.data()[i]=uint8_t(37*i+i/3);}
} // namespace

TEST_CASE("task_color_and_channel_simd_never_touch_protected_tail_pages",
          "[task][simd-dispatch][memory][guard-page]") {
    struct Operation{kernels::color::PixelTransform* run;int source_channels,destination_channels;};
    const Operation ops[]={
      {kernels::channel::CopyMonoPixels,1,1},{kernels::channel::CopyTriplePixels,3,3},{kernels::channel::CopyQuadPixels,4,4},
      {kernels::channel::ReplicateMonoToTriple,1,3},{kernels::channel::ReplicateMonoToQuad,1,4},
      {kernels::channel::AppendOpaqueAlpha,3,4},{kernels::channel::ReverseTriple,3,3},
      {kernels::channel::ReverseQuadColor,4,4},{kernels::channel::DropAlpha,4,3},
      {kernels::channel::ReverseAndDropAlpha,4,3},{kernels::channel::LumaFromRgb,3,1},
      {kernels::channel::LumaFromBgr,3,1},{kernels::channel::LumaFromRgba,4,1},{kernels::channel::LumaFromBgra,4,1},
      {kernels::color::RgbToYCrCb,3,3},{kernels::color::BgrToYCrCb,3,3},
      {kernels::color::RgbToYuv,3,3},{kernels::color::BgrToYuv,3,3},
      {kernels::color::RgbToXyz,3,3},{kernels::color::BgrToXyz,3,3},
      {kernels::color::RgbToHsv,3,3},{kernels::color::BgrToHsv,3,3},
      {kernels::color::RgbToHsvFull,3,3},{kernels::color::BgrToHsvFull,3,3},
      {kernels::color::RgbToBgr555,3,2},{kernels::color::BgrToBgr555,3,2},
      {kernels::color::RgbToBgr565,3,2},{kernels::color::BgrToBgr565,3,2}};
    for(uint32_t mask:BoundaryMasks()){
        const ScopedCpuFeatureMask features(mask);
        for(size_t count:{size_t(1),size_t(7),size_t(8),size_t(9),size_t(15),size_t(16),size_t(17),size_t(31),size_t(32),size_t(33),size_t(255),size_t(256),size_t(257)}){
            for(size_t op=0;op<sizeof(ops)/sizeof(ops[0]);++op){
                const auto& operation=ops[op];
                BoundaryBuffer source(count*operation.source_channels),output(count*operation.destination_channels);
                Pattern(source);const std::vector<uint8_t> before(source.data(),source.data()+source.size());
                CAPTURE(mask,count,op);
                operation.run(source.data(),output.data(),count);
                REQUIRE(std::equal(before.begin(),before.end(),source.data()));
            }
        }
    }
}

TEST_CASE("task_tensor_simd_never_touch_protected_tail_pages",
          "[task][simd-dispatch][memory][guard-page]") {
    const float mean[4]={-7.25f,113.5f,.125f,255.f},scale[4]={-.25f,.0625f,.5f,.125f};
    const int channels[]={1,3,3,4,1,3};
    const int stored[]={1,3,3,4,4,4};
    for(uint32_t mask:BoundaryMasks()){
        const ScopedCpuFeatureMask features(mask);
        for(size_t count:{size_t(1),size_t(7),size_t(8),size_t(9),size_t(15),size_t(16),size_t(17),size_t(31),size_t(32),size_t(33),size_t(255),size_t(256),size_t(257)}){
            for(int operation=0;operation<6;++operation){
                BoundaryBuffer source(count*channels[operation]),output(count*stored[operation]*sizeof(float));
                Pattern(source);auto* out=reinterpret_cast<float*>(output.data());
                CAPTURE(mask,count,operation);
                switch(operation){
                  case 0:kernels::tensor::InterleavedMono(source.data(),out,mean,scale,count);break;
                  case 1:kernels::tensor::InterleavedTriple(source.data(),out,mean,scale,count);break;
                  case 2:kernels::tensor::PlanarTriple(source.data(),out,count,mean,scale,count);break;
                  case 3:kernels::tensor::InterleavedQuad(source.data(),out,mean,scale,count);break;
                  case 4:kernels::tensor::QuadFromMono(source.data(),out,mean,scale,count);break;
                  case 5:kernels::tensor::QuadFromTriple(source.data(),out,mean,scale,count);break;
                }
                REQUIRE(std::isfinite(out[0]));
            }
        }
    }
}

TEST_CASE("task_sampling_simd_right_border_is_inside_last_source_row",
          "[task][simd-dispatch][memory][guard-page]") {
    kernels::sampling::Sampler* nearest[]={kernels::sampling::NearestMono,kernels::sampling::NearestTriple,kernels::sampling::NearestQuad};
    kernels::sampling::Sampler* linear[]={kernels::sampling::BilinearMono,kernels::sampling::BilinearTriple,kernels::sampling::BilinearQuad};
    for(uint32_t mask:BoundaryMasks()){
        const ScopedCpuFeatureMask features(mask);
        for(int format=0;format<3;++format){
            const int channels=format==0?1:format+2;
            for(size_t width:{size_t(1),size_t(3),size_t(4),size_t(5),size_t(7),size_t(8),size_t(17)}){
                BoundaryBuffer source(width*channels);Pattern(source);
                for(size_t count:{size_t(1),size_t(7),size_t(8),size_t(9),size_t(17),size_t(257)}){
                    for(bool bilinear:{false,true})for(bool right:{false,true}){
                        BoundaryBuffer output(count*channels);
                        Point line[2]={{right?float(width-1):-1.f,0.f},{0.f,0.f}};
                        CAPTURE(mask,channels,width,count,bilinear,right);
                        (bilinear?linear[format]:nearest[format])(source.data(),output.data(),line,0,count,count,width,1,width*channels);
                        const size_t pixel=right?width-1:0;
                        bool equal=true;
                        for(size_t i=0;i<count;++i)for(int c=0;c<channels;++c)
                            equal=equal&&output.data()[i*channels+c]==source.data()[pixel*channels+c];
                        REQUIRE(equal);
                    }
                }
            }
        }
    }
}

TEST_CASE("image_geometry_simd_reads_only_owned_pixel_bytes",
          "[image][simd-dispatch][memory][guard-page]") {
    for(uint32_t mask:BoundaryMasks()){
        const ScopedCpuFeatureMask features(mask);
        for(int channels:{1,3,4})for(int width:{1,3,4,7,8,9,15,16,17,31,32,33}){
            BoundaryBuffer storage(size_t(width)*channels);Pattern(storage);
            auto image=inspirecv::Image::Create(width,1,channels,storage.data(),false);
            CAPTURE(mask,channels,width);
            REQUIRE(image.Rotate90().Width()==1);
            REQUIRE(image.Rotate180().Width()==width);
            REQUIRE(image.Rotate270().Width()==1);
            REQUIRE(image.FlipHorizontal().Width()==width);
            REQUIRE(image.FlipVertical().Width()==width);
            REQUIRE(image.MeanChannels().Channels()==1);
            if(channels==3) REQUIRE(image.SwapRB().Channels()==3);
#if !defined(INSPIRECV_BACKEND_OPENCV) && \
    (!defined(INSPIRECV_TEST_BACKEND_OPENCV) || !INSPIRECV_TEST_BACKEND_OPENCV)
            // Four-channel alpha-preserving SwapRB is an OKCV extension.
            // The optional OpenCV backend retains its historical C4->C3 result.
            if(channels==4) REQUIRE(image.SwapRB().Channels()==4);
#endif
            if(channels==3) REQUIRE(image.ToGray().Channels()==1);
        }
    }
}
#endif
