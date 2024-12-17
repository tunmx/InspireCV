#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/poll.h>
#include <time.h>
#include <unistd.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <termios.h>
#include <linux/videodev2.h>

#include "rk_debug.h"
#include "rk_defines.h"
#include "rk_mpi_adec.h"
#include "rk_mpi_aenc.h"
#include "rk_mpi_ai.h"
#include "rk_mpi_ao.h"
#include "rk_mpi_avs.h"
#include "rk_mpi_cal.h"
#include "rk_mpi_ivs.h"
#include "rk_mpi_mb.h"
#include "rk_mpi_rgn.h"
#include "rk_mpi_sys.h"
#include "rk_mpi_tde.h"
#include "rk_mpi_vdec.h"
#include "rk_mpi_venc.h"
#include "rk_mpi_vi.h"
#include "rk_mpi_vo.h"
#include "rk_mpi_vpss.h"
#include "rga/RgaUtils.h"
#include "rga/im2d.hpp"
#include "dma_alloc.h"
#include "solex_smart_faucet_api.h"


#define WIDTH 640
#define HEIGHT 480
#define DEV_NAME "/dev/video0"
#define BUFFER_SIZE 5//设置1个缓冲帧
#define PIX_SIZE (WIDTH * HEIGHT * 3) >> 1
char* uvc_buf;
int camera_fd = -1;
int bytesused;

#define WRITE_TO_FILE 0
#define AI 1

static FILE *fp = NULL;
static bool quit = false;
static RK_S32 g_s32FrameCnt = -1;
#if AI
static SL_SmartFaucetAIRunnerContext *ctx = NULL;
#endif

static void sigterm_handler(int sig) {
	fprintf(stderr, "signal %d\n", sig);
	quit = true;
}

// 获取每像素的位数
int get_bpp_from_format1(int format) {
    switch (format) {
        case RK_FORMAT_YUYV_422:
            return 2; // YUYV 422 格式的每像素位数为 16
        case RK_FORMAT_RGB_888:
            return 3; // RGB 888 格式的每像素位数为 24
        default:
            printf("未知的图像格式\n");
            return -1;
    }
}
struct dma_buf_sync {
	__u64 flags;
};

#define DMA_BUF_BASE		'b'
#define DMA_BUF_IOCTL_SYNC	_IOW(DMA_BUF_BASE, 0, struct dma_buf_sync)

#define CMA_HEAP_SIZE	1024 * 1024
int dma_sync_device_to_cpu(int fd) {
    struct dma_buf_sync sync = {0};

    sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
}

int dma_sync_cpu_to_device(int fd) {
    struct dma_buf_sync sync = {0};

    sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
}
int dma_buf_alloc(const char *path, size_t size, int *fd, void **va) {
    int ret;
    int prot;
    void *mmap_va;
    int dma_heap_fd = -1;
    struct dma_heap_allocation_data buf_data;

    /* open dma_heap fd */
    dma_heap_fd = open(path, O_RDWR);
    if (dma_heap_fd < 0) {
        printf("open %s fail!\n", path);
        return dma_heap_fd;
    }

    /* alloc buffer */
    memset(&buf_data, 0x0, sizeof(struct dma_heap_allocation_data));

    buf_data.len = size;
    buf_data.fd_flags = O_CLOEXEC | O_RDWR;
    ret = ioctl(dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &buf_data);
    if (ret < 0) {
        printf("RK_DMA_HEAP_ALLOC_BUFFER failed\n");
        return ret;
    }

    /* mmap va */
    if (fcntl(buf_data.fd, F_GETFL) & O_RDWR)
        prot = PROT_READ | PROT_WRITE;
    else
        prot = PROT_READ;

    /* mmap contiguors buffer to user */
    mmap_va = (void *)mmap(NULL, buf_data.len, prot, MAP_SHARED, buf_data.fd, 0);
    if (mmap_va == MAP_FAILED) {
        printf("mmap failed: %s\n", strerror(errno));
        return -errno;
    }

    *va = mmap_va;
    *fd = buf_data.fd;

    close(dma_heap_fd);

    return 0;
}

void dma_buf_free(size_t size, int *fd, void *va) {
    int len;

    len =  size;
    munmap(va, len);

    close(*fd);
    *fd = -1;
}

void  capture_test()
{
	system("echo \"host\" > /sys/devices/platform/ff3e0000.usb2-phy/otg_mode");
	int ret,i;

	struct v4l2_format fmt;
	struct v4l2_fmtdesc fmtdesc;
	struct v4l2_capability cap;
	struct v4l2_requestbuffers reqbuf;
	struct v4l2_buffer buf;
	struct v4l2_streamparm *setfps;
	unsigned char* buffers[BUFFER_SIZE];
	int width = WIDTH;
	int height = HEIGHT;
	#define MAX_FPS 30

	fd_set fds;
	struct timeval tv;
	int camera_index;


	for(i = 0; i < BUFFER_SIZE; i++){
		buffers[i] = malloc(PIX_SIZE);
		if(buffers[i] == NULL)
			printf("can not mallo for buffers[%d]\n", i);
	}

	//打开视频设备
	camera_fd = open(DEV_NAME, O_RDWR);
	if (camera_fd == -1) {
		printf("Failed to open video device\n");
		return (void *)(-1);
	}
	printf("[step]open UVC 0ful\n");

	//获取设备支持的格式
	memset(&fmtdesc, 0, sizeof(fmtdesc));
	fmtdesc.index = 0;
	fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	while((ret = ioctl(camera_fd, VIDIOC_ENUM_FMT, &fmtdesc)) == 0){
		fmtdesc.index++;
		printf("{ pixelformat = \'%c%c%c%c\', description = \'%s\' }\n",\
		fmtdesc.pixelformat & 0xFF, (fmtdesc.pixelformat >> 8) & 0xFF, \
		(fmtdesc.pixelformat >> 16) & 0xFF, (fmtdesc.pixelformat >> 24) & 0xFF, \
		fmtdesc.description);
	}
	//查询设备能力,capabilities是用掩码表示
	ret = ioctl(camera_fd, VIDIOC_QUERYCAP, &cap);
	if(ret < 0)
	{
		printf("get vidieo capability error,error code: %d \n", ret);
		goto exit;
	}
	printf("{ Capability: driver:\'%s\', card:\'%s\',buf_info:\'%s\',version:%d,capabilities:0x%x}\n",\
	cap.driver,cap.card,cap.bus_info,cap.version,cap.capabilities);

	memset(&fmt, 0, sizeof(fmt));
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.width = width;	// 设置宽度
	fmt.fmt.pix.height = height;  // 设置高度
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;  // 设置像素格式
	ret = ioctl(camera_fd, VIDIOC_S_FMT, &fmt);
	if(ret < 0){
		printf("SET VIDEO FORMAT ERROR, err=%d\n", ret);
		goto exit;
	}
	printf("{ Format width: %d, height:%d}\n", fmt.fmt.pix.width, fmt.fmt.pix.height);

	//设置视频流信息，关于视频fps
	setfps = (struct v4l2_streamparm *)calloc(1,sizeof(struct v4l2_streamparm));
	memset(setfps,0,sizeof(struct v4l2_streamparm));
	setfps->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	setfps->parm.capture.timeperframe.numerator = 1;
	setfps->parm.capture.timeperframe.denominator = MAX_FPS;
	if(ioctl(camera_fd, VIDIOC_S_PARM,setfps) < 0)
	{
		printf("set VIDIOC_S_PARM error");
		goto exit;
	}
	printf("[step]set stream fps 0ful\n");

	//申请视频缓冲区
	memset(&reqbuf, 0, sizeof(reqbuf));
	reqbuf.count = BUFFER_SIZE;
	reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	reqbuf.memory = V4L2_MEMORY_MMAP;
	if(ioctl(camera_fd, VIDIOC_REQBUFS, &reqbuf) < 0){
		printf("set VIDIOC_REQBUFS error");
		goto exit;
	}
	printf("[step]request buffer 0ful\n");

	//获取缓冲帧的地址，长度：
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.index = 0;
	buf.memory = V4L2_MEMORY_MMAP;
	if(ioctl(camera_fd,VIDIOC_QUERYBUF,&buf) < 0){
		printf("set VIDIOC_QUERYBUF error");
		goto exit;
	}
	printf("[step]request buffer add and lens 0ful\n");

	//获取缓冲帧的地址,并以mmap共享内存的方式将缓冲帧与应用层共享
	for (i = 0; i < BUFFER_SIZE; ++i) {
		memset(&buf, 0, sizeof(buf));
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = i;
		if(ioctl(camera_fd, VIDIOC_QUERYBUF, &buf) < 0 ){
			printf("set VIDIOC_QUERYBUF error");
			goto exit;
		}
		printf("{ Querybuffer length:%d, m.offset: %d}\n",buf.length,buf.m.offset);
		buffers[i] = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, camera_fd, buf.m.offset);
	}
	printf("[step]mmap buffer 0ful\n");

	//将缓冲帧放入缓冲队列
	for (i = 0; i < BUFFER_SIZE; ++i) {
		memset(&buf, 0, sizeof(buf));
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = i;
		if(ioctl(camera_fd, VIDIOC_QBUF, &buf) < 0){
			printf("set VIDIOC_QBUF error");
			goto ummp;
		}
	}

	//启动视频流
	enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if(ioctl(camera_fd, VIDIOC_STREAMON, &type) < 0){
		printf("set VIDIOC_QUERYBUF error");
		goto ummp;
	}
		
	printf("[step]stream on 0ful\n");

	//设置超时时间
	for(camera_index = 0;camera_index < MAX_FPS;camera_index++){
		FD_ZERO(&fds);
		FD_SET(camera_fd, &fds);
		tv.tv_sec = 2;
		tv.tv_usec = 0;
		//以select方式监听数据，超时2s
		ret = select(camera_fd+1,&fds,NULL,NULL,&tv);
		if(ret == -1){
			printf("select\n");
			goto ummp;
		}
		else if(ret == 0){
			printf("select time out\n");
			goto ummp;
		}
	}
	rga_buffer_t src_img,dst_img,dst_rotate_img;
	rga_buffer_handle_t src_handle,dst_handle,dst_rotate_handle;
	char *dst_buf, *dst_rorate_buf;
	memset(&src_img, 0, sizeof(src_img));
    memset(&dst_img, 0, sizeof(dst_img));

	int src_buf_size, dst_buf_size;
	src_buf_size = WIDTH * HEIGHT * get_bpp_from_format1(RK_FORMAT_YUYV_422);
	dst_buf_size = WIDTH * HEIGHT *get_bpp_from_format1(RK_FORMAT_RGB_888);
	im_handle_param_t src_para = {0};
	im_handle_param_t dst_para = {0};
	im_handle_param_t dst_rotate_para = {0};
	src_para.width = WIDTH;
	src_para.height = HEIGHT;
	src_para.format = RK_FORMAT_YUYV_422;
	dst_para.width = WIDTH;
	dst_para.height = HEIGHT;
	dst_para.format = RK_FORMAT_RGB_888;
	dst_rotate_para.width = WIDTH;
	dst_rotate_para.height = HEIGHT;
	dst_rotate_para.format = RK_FORMAT_RGB_888;

	int src_dma_fd, dst_dma_fd,dst_dma_rotate_fd;
	ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH,src_buf_size,&src_dma_fd,(void **)&uvc_buf);
	if (ret < 0) {
        printf("alloc src CMA buffer failed!\n");
        return -1;
    }
	ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH,dst_buf_size,&dst_dma_fd,(void **)&dst_buf);
	if (ret < 0) {
        printf("alloc dst CMA buffer failed!\n");
        return -1;
    }
	ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH,dst_buf_size,&dst_dma_rotate_fd,(void **)&dst_rorate_buf);
	if (ret < 0) {
		printf("alloc dst rotate CMA buffer failed!\n");
		return -1;
	}

	printf("capture...\n");
	memset(uvc_buf, 0, src_buf_size);
	struct timeval start, end;
	long mtime, seconds, useconds;
	int loopCount = 0;
	// FILE *fd1 = NULL;
	// fd1 = fopen("uvc.yuv", "wb");
	// 	if (fp == NULL) {
	// 		printf("open file errpr!\n");
	// 		return -1;
	// 	}
	while(1) {
		printf("----loopCount:%d-------\n",loopCount);
		gettimeofday(&start, NULL);
		memset(&buf, 0, sizeof(buf));
		printf("buf set ok\n");
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		ioctl(camera_fd, VIDIOC_DQBUF, &buf);

		memcpy(uvc_buf, buffers[buf.index], buf.bytesused);
		// fwrite(uvc_buf,src_buf_size,1,fd1);
		//memset(dst_buf, 0x33, dst_buf_size);
		// 获取采集数据结束的时间，并计算耗时
		gettimeofday(&end, NULL);
		seconds  = end.tv_sec  - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;
		mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
		printf("采集数据耗时: %ld 毫秒\n", mtime);

		gettimeofday(&start, NULL);
		dma_sync_cpu_to_device(src_dma_fd);
		dma_sync_cpu_to_device(dst_dma_fd);
		dma_sync_cpu_to_device(dst_dma_rotate_fd);
		src_handle = importbuffer_fd(src_dma_fd,&src_para);
		dst_handle = importbuffer_fd(dst_dma_fd,&dst_para);
		dst_rotate_handle = importbuffer_fd(dst_dma_rotate_fd,&dst_rotate_para);
		if (src_handle == 0 || dst_handle == 0 || dst_rotate_handle == 0) {
			printf("import dma_fd error!\n");
			goto release_buffer;
		}
		src_img = wrapbuffer_handle(src_handle,WIDTH,HEIGHT,RK_FORMAT_YUYV_422);
		dst_img = wrapbuffer_handle(dst_handle,WIDTH,HEIGHT,RK_FORMAT_RGB_888);
		dst_rotate_img = wrapbuffer_handle(dst_rotate_handle,WIDTH,HEIGHT,RK_FORMAT_RGB_888);

		ret = imcvtcolor(src_img,dst_img,RK_FORMAT_YUYV_422,RK_FORMAT_RGB_888);
		if (ret == IM_STATUS_SUCCESS) {
			printf("%s running success!\n", "imcvtcolor");
		} else {
			printf("%s running failed, %s\n", "imcvtcolor", imStrError((IM_STATUS)ret));
			goto release_buffer;
		}
		ret = imrotate(dst_img,dst_rotate_img,IM_HAL_TRANSFORM_ROT_180);
		if (ret == IM_STATUS_SUCCESS) {
			printf("%s running success!\n", "imrotate");
		} else {
			printf("%s running failed, %s\n", "imrotate", imStrError((IM_STATUS)ret));
			goto release_buffer;
		}

		// 获取转换格式结束的时间，并计算耗时
		gettimeofday(&end, NULL);
		seconds  = end.tv_sec  - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;
		mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
		printf("转换格式耗时: %ld 毫秒\n", mtime);

#if AI
		gettimeofday(&start, NULL);
		SRESULT ret1;
		// SL_CameraStream *stream = SL_CreateCameraStream();
		SL_ThingResult thingResult = {0};
		int save_data = 1;
		SL_ImageBuffer sl_img = {0};
		sl_img.fd = dst_dma_rotate_fd;
		sl_img.width = WIDTH;
		sl_img.height = HEIGHT;
		sl_img.width_stride = 1;
		sl_img.height_stride = 1;
		sl_img.size = dst_buf_size;
		sl_img.virt_addr = dst_rorate_buf;
		sl_img.format = SL_IMAGE_FORMAT_RGB888;
		printf("dst_dma_fd:%d\n",dst_dma_rotate_fd);
		// SL_CameraStreamSetRotationMode(stream, CAMERA_ROTATION_0);
		// SL_CameraStreamSetStreamFormat(stream, STREAM_RGB);
		// SL_CameraStreamSetData(stream, dst_buf, WIDTH, HEIGHT);
		// ret1= SL_SmartFaucetAIRunnerContextThingDetection(ctx, stream, &thingResult, save_data);
		// if (ret1 != 0)
		// {
		// 	printf("detect error!\r\n");
		// }
		ret1= SL_SmartFaucetAIRunnerContextThingDetection(ctx,&sl_img,&thingResult,save_data);
		if (ret1 != 0)
		{
			printf("detect error!\r\n");
		}
		printf("ronfull thingResult.category=%d, thingResult.water_level=%d\r\n", thingResult.category, thingResult.water_level);
		// SL_ReleaseCameraStream(stream);
		// 获取算法结束的时间，并计算耗时
		gettimeofday(&end, NULL);
		seconds  = end.tv_sec  - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;
		mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
		printf("算法识别耗时: %ld 毫秒\n", mtime);
#endif

#if WRITE_TO_FILE
		gettimeofday(&start, NULL);
		fwrite(dst_rorate_buf, dst_buf_size, 1, fp);
		// char filename[100];
		// sprintf(filename, "frame-%03d.rgb", loopCount);
		// FILE *imageFile = fopen(filename, "wb");
		// if (imageFile == NULL) {
		// 	printf("无法打开图片文件\n");
		// 	return -1;
		// }
		// fwrite(dst_buf, dst_buf_size, 1, imageFile);
		// fclose(imageFile);
		// 获取写文件结束的时间，并计算耗时
		gettimeofday(&end, NULL);
		seconds  = end.tv_sec  - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;
		mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
		printf("写文件耗时: %ld 毫秒\n", mtime);
#endif

		dma_sync_device_to_cpu(src_dma_fd);
		dma_sync_device_to_cpu(dst_dma_fd);
		dma_sync_device_to_cpu(dst_dma_rotate_fd);
		gettimeofday(&start, NULL);
		if (src_handle)
			releasebuffer_handle(src_handle);
		if (dst_handle)
			releasebuffer_handle(dst_handle);
		if (dst_rotate_handle)
			releasebuffer_handle(dst_rotate_handle);

		bytesused = buf.bytesused;
		ioctl(camera_fd, VIDIOC_QBUF, &buf);
		// 获取释放结束的时间，并计算耗时
		gettimeofday(&end, NULL);
		seconds  = end.tv_sec  - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;
		mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
		printf("释放操作耗时: %ld 毫秒\n", mtime);
		loopCount++;
	}
	printf("bytesused: %d, buf.bytesused:%d\n", bytesused, buf.bytesused);
	
	fclose(fp);
	

	printf("[step]write data 0ful\n");

	// 停止视频采集并释放资源
	ioctl(camera_fd, VIDIOC_STREAMOFF, &type);
	for (i = 0; i < BUFFER_SIZE; ++i) {
		munmap(buffers[i], buf.length);
	}

	printf("[step]stop stream 0ful\n");

	close(camera_fd);
	return (void *)(0);
release_buffer:
    if (src_handle)
        releasebuffer_handle(src_handle);
    if (dst_handle)
        releasebuffer_handle(dst_handle);
	if (dst_rotate_handle)
			releasebuffer_handle(dst_rotate_handle);

	dma_buf_free(src_buf_size, &src_dma_fd, uvc_buf);
    dma_buf_free(dst_buf_size, &dst_dma_fd, dst_buf);
	dma_buf_free(dst_buf_size,&dst_dma_rotate_fd,dst_rorate_buf);

    if (dst_buf)
        free(dst_buf);
	if (dst_rorate_buf)
		free(dst_rorate_buf);
ummp:
	for (i = 0; i < BUFFER_SIZE; ++i) {
		munmap(buffers[i], buf.length);
	}
exit:
	close(camera_fd);
	return (void *)(-1);
}

int main()
{
    signal(SIGINT, sigterm_handler);

#if WRITE_TO_FILE
        // 如果宏定义为真，则将pData写入文件
        fp = fopen("output.rgb", "wb");
		if (fp == NULL) {
			printf("open file errpr!\n");
			return -1;
		}
#endif

    RK_S32 s32Ret = RK_FAILURE;
    RK_S32 s32chnlId = 0;
    MPP_CHN_S stSrcChn, stvpssChn, stvencChn;

    if (RK_MPI_SYS_Init() != RK_SUCCESS) {
		RK_LOGE("RK_MPI_SYS_Init failure");
		return -1;
	}
#if AI == 1
    SL_STSDKVersion version = {0};
    SL_GetSDKVersionInfo(&version);
    printf("solex ai closestool version: v%d.%d.%d\n", version.major, version.minor, version.patch);
    SL_SetSDKLoggerLevel(SL_INFO);
    ctx = SL_CreateSmartFaucetAIRunnerContext("/oem/usr/lib/STPackEndSide");
    printf("main APP:date:%s,time:%s\r\n", __DATE__, __TIME__);
#endif
    pthread_t main_thread;
	pthread_create(&main_thread, NULL, capture_test, NULL);
    while (!quit) {
		usleep(50000);
	}
	fclose(fp);
}