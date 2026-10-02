// Supplemental same-host baseline/candidate benchmark. Only public APIs are
// used, so this exact source also links against pre-SIMD InspireCV builds.
#include <inspirecv/inspirecv.h>
#include <inspirecv/task/task.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;
using namespace inspirecv;
using namespace inspirecv::task;
volatile uint64_t sink=0;
struct Options {
    std::string report="simd_coverage.csv",suite="all",operation;
    int samples=7, max_width=1920;
    double min_ms=15;
    bool allow_missing_hsvfull=false, skip_c4_swap=false, skip_padded_i420=false;
};
Options Parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if(key=="--allow-missing-hsvfull") {o.allow_missing_hsvfull=true;continue;}
        if(key=="--skip-c4-swap") {o.skip_c4_swap=true;continue;}
        if(key=="--skip-padded-i420") {o.skip_padded_i420=true;continue;}
        if(i+1==argc) throw std::runtime_error("missing option value");
        const std::string value=argv[++i];
        if(key=="--report") o.report=value;
        else if(key=="--suite") o.suite=value;
        else if(key=="--operation") o.operation=value;
        else if(key=="--samples") o.samples=std::stoi(value);
        else if(key=="--min-ms") o.min_ms=std::stod(value);
        else if(key=="--max-width") o.max_width=std::stoi(value);
        else throw std::runtime_error("unknown option: "+key);
    }
    if(o.samples<3||o.samples>101||o.min_ms<=0||o.min_ms>1000||o.max_width<=0 ||
       (o.suite!="all"&&o.suite!="image"&&o.suite!="task"))
        throw std::runtime_error("invalid benchmark options");
    return o;
}
uint64_t Hash(const void* data,size_t size) {
    auto p=static_cast<const uint8_t*>(data);
    uint64_t h=UINT64_C(1469598103934665603);
    for(size_t i=0;i<size;++i) h=(h^p[i])*UINT64_C(1099511628211);
    return h;
}
struct Timing {double p50=0,p95=0,minimum=0,maximum=0;int loops=0;};
Timing Measure(const std::function<void()>& run,const std::function<void()>& consume,const Options& o) {
    run();run();consume();
    int loops=1;
    for(;;) {
        const auto start=Clock::now();
        for(int i=0;i<loops;++i)run();
        const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        consume();
        if(elapsed>=o.min_ms||loops>=65536) break;
        const double gain=std::max(2.0,std::min(8.0,o.min_ms/std::max(.001,elapsed)));
        loops=std::min(65536,std::max(loops+1,int(loops*gain)));
    }
    std::vector<double> times;
    for(int sample=0;sample<o.samples;++sample) {
        const auto start=Clock::now();
        for(int i=0;i<loops;++i) run();
        const double elapsed=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
        consume();
        times.push_back(elapsed/loops);
    }
    std::sort(times.begin(),times.end());
    return {times[times.size()/2],times[size_t(std::ceil(.95*times.size()))-1],times.front(),times.back(),loops};
}
std::string CsvField(const char* value) {
    std::string escaped="\"";
    for(const char* p=value;*p;++p) {if(*p=='"')escaped+='"';escaped+=*p;}
    return escaped+'"';
}
struct Shape {int w,h;};
const Shape sizes[]={{112,112},{224,224},{513,257},{640,480},{1920,1080}};
class Report {
public:
    Report(const Options& o):options(o),out(o.report) {
        if(!out)throw std::runtime_error("cannot create report");
        out<<"family,operation,dtype,channels,source_width,source_height,output_width,output_height,source_stride,layout,timing_scope,cpu_disable,status,p50_us,p95_us,min_us,max_us,batch_iterations,samples,output_bytes,checksum,note\n";
    }
    void Row(const char* family,const std::string& op,const char* type,int channels,
             Shape source,Shape dest,size_t stride,const char* layout,const char* scope,
             int status,const Timing& time,size_t bytes,uint64_t checksum,const char* note="") {
        const char* mask=std::getenv("INSPIRECV_INTERNAL_CPU_DISABLE");
        out<<family<<','<<op<<','<<type<<','<<channels<<','<<source.w<<','<<source.h<<','
           <<dest.w<<','<<dest.h<<','<<stride<<','<<layout<<','<<scope<<','<<CsvField(mask?mask:"default")<<','
           <<status<<','<<std::fixed<<std::setprecision(6)<<time.p50<<','<<time.p95<<','<<time.minimum<<','
           <<time.maximum<<','<<time.loops<<','<<options.samples<<','<<bytes<<','<<std::hex<<checksum<<std::dec<<','<<CsvField(note)<<'\n';
        out.flush();
        std::cout<<family<<'/'<<op<<' '<<type<<" C"<<channels<<' '<<source.w<<'x'<<source.h
                 <<" -> "<<dest.w<<'x'<<dest.h<<" status="<<status<<" p50_us="<<time.p50<<'\n';
    }
    Options options;
private:std::ofstream out;
};

template<class T>void ImageCase(Report& report,const std::string& op,const ImageT<T>& source,
                                const std::function<ImageT<T>()>& operation) {
    if(!report.options.operation.empty() && report.options.operation!=op)return;
    ImageT<T> output;
    const auto run=[&]{output=operation();};
    const auto consume=[&]{if(!output.Empty())sink+=uint64_t(output.Width())+uint64_t(output.Height())+uint64_t(reinterpret_cast<const uint8_t*>(output.Data())[0]);};
    const auto timing=Measure(run,consume,report.options);
    const size_t bytes=size_t(output.Width())*output.Height()*output.Channels()*sizeof(T);
    report.Row("image",op,std::is_same<T,float>::value?"f32":"u8",source.Channels(),
      {source.Width(),source.Height()},{output.Width(),output.Height()},size_t(source.Width())*source.Channels()*sizeof(T),
      "HWC","public_api_allocation",0,timing,bytes,Hash(output.Data(),bytes));
}
template<class T>void ImageCases(Report& report) {
    for(const auto size:sizes) {
        if(size.w>report.options.max_width)continue;
        for(int channels:{1,3,4}) {
            const size_t elements=size_t(size.w)*size.h*channels;
            std::vector<T> pixels(elements),other(elements);
            std::vector<uint8_t> mask(size_t(size.w)*size.h);
            for(size_t i=0;i<elements;++i) {
                pixels[i]=T((i*37+i/11*19)%256);
                other[i]=T((i*13+i/7*43)%256);
                if(std::is_same<T,float>::value){pixels[i]=pixels[i]*T(.5)-T(16);other[i]+=T(.25);}
            }
            for(size_t i=0;i<mask.size();++i)mask[i]=uint8_t(i*71+i/13);
            const auto image=ImageT<T>::Create(size.w,size.h,channels,pixels.data(),false);
            const auto second=ImageT<T>::Create(size.w,size.h,channels,other.data(),false);
            const auto alpha=Image::Create(size.w,size.h,1,mask.data(),false);
            auto add=[&](const std::string& name,const std::function<ImageT<T>()>& fn){ImageCase<T>(report,name,image,fn);};
            add("clone",[&]{return image.Clone();});
            add("rotate90",[&]{return image.Rotate90();});
            add("rotate180",[&]{return image.Rotate180();});
            add("rotate270",[&]{return image.Rotate270();});
            add("flip_horizontal",[&]{return image.FlipHorizontal();});
            add("flip_vertical",[&]{return image.FlipVertical();});
            // A genuine rotation/shear exercises the general affine route;
            // scale-only transforms take a separate optimized implementation.
            const float center_x=(size.w-1)*.5f,center_y=(size.h-1)*.5f;
            const float cosine=.9659258f,sine=.258819f;
            const auto affine=TransformMatrix::Create(cosine,sine,
              center_x-cosine*center_x-sine*center_y,-sine,cosine,
              center_y+sine*center_x-cosine*center_y);
            add("warp_affine_general",[&]{return image.WarpAffine(affine,size.w,size.h);});
            for(bool linear:{false,true}) {
                const std::string method=linear?"resize_linear":"resize_nearest";
                add(method+"_half",[&]{return image.Resize(std::max(1,size.w/2),std::max(1,size.h/2),linear);});
                if(size.w>=640) add(method+"_224",[&]{return image.Resize(224,224,linear);});
                if(size.w==1920) add(method+"_4k",[&]{return image.Resize(3840,2160,linear);});
            }
            add("absdiff",[&]{return image.AbsDiff(second);});
            add("blend",[&]{return image.Blend(second,alpha);});
            add("mean_channels",[&]{return image.MeanChannels();});
            add("mul",[&]{return image.Mul(.75);});
            add("add",[&]{return image.Add(7.5);});
            add("crop",[&]{return image.Crop(inspirecv::Rect<int>(1,1,size.w-2,size.h-2));});
            const std::vector<double> pad_color = channels==1?std::vector<double>{11}:channels==3?std::vector<double>{11,23,47}:std::vector<double>{11,23,47,71};
            add("pad",[&]{return image.Pad(3,5,7,9,pad_color);});
            if(channels>=3) {
                if(channels==4 && report.options.skip_c4_swap) {
                  if(report.options.operation.empty() || report.options.operation=="swap_rb")
                    report.Row("image","swap_rb",std::is_same<T,float>::value?"f32":"u8",channels,size,size,
                      size_t(size.w)*channels*sizeof(T),"HWC","public_api_allocation",-1,Timing{},0,0,
                      "explicit_skip_baseline_c4_swap_unsupported");
                } else add("swap_rb",[&]{return image.SwapRB();});
            }
            if(channels==3)add("to_gray",[&]{return image.ToGray();});
            if(channels==1||channels==3) {
                for(int kernel:{3,5})add("gaussian"+std::to_string(kernel),[&]{return image.GaussianBlur(kernel,0);});
            }
            if(channels==1) {
                add("threshold",[&]{return image.Threshold(127.5,255,0);});
                for(int kernel:{3,15,16}) {
                    add("erode"+std::to_string(kernel),[&]{return image.Erode(kernel,1);});
                    add("dilate"+std::to_string(kernel),[&]{return image.Dilate(kernel,1);});
                }
            }
        }
    }
}
struct Conversion {PixelFormat from,to;int in,out;const char* name;};
const Conversion colors[]={
 {PixelFormat::kRgb,PixelFormat::kYCrCb,3,3,"rgb_ycrcb"},{PixelFormat::kBgr,PixelFormat::kYCrCb,3,3,"bgr_ycrcb"},
 {PixelFormat::kRgb,PixelFormat::kYuv,3,3,"rgb_yuv"},{PixelFormat::kBgr,PixelFormat::kYuv,3,3,"bgr_yuv"},
 {PixelFormat::kRgb,PixelFormat::kXyz,3,3,"rgb_xyz"},{PixelFormat::kBgr,PixelFormat::kXyz,3,3,"bgr_xyz"},
 {PixelFormat::kRgb,PixelFormat::kHsv,3,3,"rgb_hsv"},{PixelFormat::kBgr,PixelFormat::kHsv,3,3,"bgr_hsv"},
 {PixelFormat::kRgb,PixelFormat::kHsvFull,3,3,"rgb_hsvfull"},{PixelFormat::kBgr,PixelFormat::kHsvFull,3,3,"bgr_hsvfull"},
 {PixelFormat::kRgb,PixelFormat::kBgr555,3,2,"rgb_bgr555"},{PixelFormat::kBgr,PixelFormat::kBgr555,3,2,"bgr_bgr555"},
 {PixelFormat::kRgb,PixelFormat::kBgr565,3,2,"rgb_bgr565"},{PixelFormat::kBgr,PixelFormat::kBgr565,3,2,"bgr_bgr565"},
 {PixelFormat::kBgr,PixelFormat::kRgb,3,3,"bgr_rgb"},{PixelFormat::kRgba,PixelFormat::kBgra,4,4,"rgba_bgra"},
 {PixelFormat::kRgba,PixelFormat::kBgr,4,3,"rgba_bgr"},{PixelFormat::kBgra,PixelFormat::kBgr,4,3,"bgra_bgr"},
 {PixelFormat::kRgb,PixelFormat::kRgba,3,4,"rgb_rgba"},{PixelFormat::kBgr,PixelFormat::kGray,3,1,"bgr_gray"},
 {PixelFormat::kRgb,PixelFormat::kGray,3,1,"rgb_gray"},{PixelFormat::kRgba,PixelFormat::kGray,4,1,"rgba_gray"},
 {PixelFormat::kBgra,PixelFormat::kGray,4,1,"bgra_gray"},{PixelFormat::kGray,PixelFormat::kBgr,1,3,"gray_bgr"},
 {PixelFormat::kGray,PixelFormat::kBgra,1,4,"gray_bgra"}};
const char* OrderName(TensorOrder order){return order==TensorOrder::kHwc?"HWC":order==TensorOrder::kChw?"CHW":"NC4HW4";}
void TaskCase(Report& report,const Conversion& conversion,Shape size,int padding,
              bool floats,TensorOrder order,int transform) {
    const bool yuv=conversion.from==PixelFormat::kNv12||conversion.from==PixelFormat::kNv21||conversion.from==PixelFormat::kI420;
    if(yuv){size.w&=~1;size.h&=~1;}
    const size_t source_stride=size.w*conversion.in+(yuv?padding*2:padding);
    const size_t source_bytes=source_stride*size.h*(yuv?3:2)/2;
    std::vector<uint8_t> input(source_bytes);
    for(size_t i=0;i<input.size();++i) input[i]=uint8_t(i*37+i/11*19);
    const Shape dest=transform?Shape{224,224}:size;
    const int stored_channels=order==TensorOrder::kChannelPacked4?4:conversion.out;
    const size_t row_elements=(order==TensorOrder::kChw?dest.w:dest.w*stored_channels)+(padding?4:0);
    const size_t plane=row_elements*dest.h+((padding&&order==TensorOrder::kChw)?8:0);
    const size_t elements=plane*(order==TensorOrder::kChw?conversion.out:1);
    std::vector<uint8_t> bytes(floats?0:elements,0xCD);
    std::vector<float> values(floats?elements:0,-9876.5f);
    TensorBuffer output;
    output.data=floats?static_cast<void*>(values.data()):static_cast<void*>(bytes.data());
    output.row_stride_bytes=row_elements*(floats?sizeof(float):1);
    output.channel_stride_bytes=order==TensorOrder::kChw?plane*sizeof(float):0;
    output.width=dest.w;output.height=dest.h;output.channels=conversion.out;
    output.element_type=floats?ElementType::kFloat32:ElementType::kUInt8;output.order=order;
    PipelineOptions options;
    options.input_format=conversion.from;options.output_format=conversion.to;
    options.backend_preference=BackendPreference::kCpu;
    options.sampling=transform==2?SamplingMode::kLinear:SamplingMode::kNearest;
    options.mean={{127.5f,113.25f,91.125f,64.f}};
    options.scale={{1.f/128.f,1.f/127.f,.03125f,.0625f}};
    Pipeline pipeline(options);
    if(transform) {
        const auto matrix=TransformMatrix::Create(float(size.w)/dest.w,0,-.25f,0,float(size.h)/dest.h,.125f);
        if(pipeline.SetTransform(matrix)!=Status::kOk)throw std::runtime_error("benchmark transform rejected");
    }
    // A packed descriptor uses the public zero-stride default. In particular,
    // old I420 interpreted explicit strides inconsistently for chroma planes.
    const size_t descriptor_stride=padding?source_stride:0;
    const RawImageView source{input.data(),descriptor_stride,size.w,size.h};
    const std::string op=std::string(conversion.name)+(floats?"_f32":"_u8")+
      (transform==1?"_nearest":transform==2?"_linear":"_identity")+(padding?"_padded":"_packed");
    if(!report.options.operation.empty() && report.options.operation!=op)return;
    if(conversion.from==PixelFormat::kI420 && padding && report.options.skip_padded_i420) {
        report.Row("task",op,floats?"u8_to_f32":"u8",conversion.out,size,dest,descriptor_stride,OrderName(order),
          "preallocated_end_to_end",-2,Timing{},0,0,"explicit_skip_baseline_padded_i420_unsafe_stride");
        return;
    }
    const Status initial=pipeline.Run(source,output);
    if(initial!=Status::kOk) {
        report.Row("task",op,floats?"u8_to_f32":"u8",conversion.out,size,dest,descriptor_stride,OrderName(order),
          "preallocated_end_to_end",int(initial),Timing{},0,0,
          report.options.allow_missing_hsvfull&&conversion.to==PixelFormat::kHsvFull
            ?"baseline_hsvfull_conversion_unsupported":"conversion_failed");
        if(!(report.options.allow_missing_hsvfull&&conversion.to==PixelFormat::kHsvFull))
            throw std::runtime_error("benchmark conversion failed: "+op);
        return;
    }
    Status status=Status::kOk;
    const auto run=[&]{status=pipeline.Run(source,output);};
    const auto consume=[&]{sink+=uint64_t(int(status))+uint64_t(source.width);};
    const auto timing=Measure(run,consume,report.options);
    if(status!=Status::kOk)throw std::runtime_error("conversion failed during timing");
    const size_t byte_count=elements*(floats?sizeof(float):1);
    report.Row("task",op,floats?"u8_to_f32":"u8",conversion.out,size,dest,descriptor_stride,OrderName(order),
      "preallocated_end_to_end",int(status),timing,byte_count,Hash(output.data,byte_count));
}
void TaskCases(Report& report) {
    for(const auto size:sizes) {
        if(size.w>report.options.max_width)continue;
        for(const auto& color:colors) for(int padding:{0,7})
            TaskCase(report,color,size,padding,false,TensorOrder::kHwc,0);
        const Conversion tensor[]={
          {PixelFormat::kGray,PixelFormat::kGray,1,1,"gray_gray"},
          {PixelFormat::kBgr,PixelFormat::kRgb,3,3,"bgr_rgb"},
          {PixelFormat::kBgra,PixelFormat::kBgra,4,4,"bgra_bgra"}};
        for(const auto& c:tensor) for(int padding:{0,7}) {
            for(auto order:{TensorOrder::kHwc,TensorOrder::kChw,TensorOrder::kChannelPacked4}) {
                if(c.out==4&&order!=TensorOrder::kHwc)continue;
                TaskCase(report,c,size,padding,true,order,0);
            }
            for(int transform:{1,2})TaskCase(report,c,size,padding,false,TensorOrder::kHwc,transform);
        }
        for(auto format:{PixelFormat::kNv12,PixelFormat::kNv21,PixelFormat::kI420}) {
            const Conversion c={format,PixelFormat::kBgr,1,3,format==PixelFormat::kNv12?"nv12_bgr":format==PixelFormat::kNv21?"nv21_bgr":"i420_bgr"};
            for(int padding:{0,7})for(int transform:{0,1})TaskCase(report,c,size,padding,false,TensorOrder::kHwc,transform);
        }
    }
}
} // namespace
int main(int argc,char** argv) {
    try {
        const auto options=Parse(argc,argv);Report report(options);
        if(options.suite=="all"||options.suite=="image"){ImageCases<uint8_t>(report);ImageCases<float>(report);}
        if(options.suite=="all"||options.suite=="task")TaskCases(report);
        std::cout<<"report="<<options.report<<" sink="<<sink<<'\n';return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
