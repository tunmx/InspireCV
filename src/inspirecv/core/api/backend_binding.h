#ifndef INSPIRECV_CORE_API_BACKEND_BINDING_H_
#define INSPIRECV_CORE_API_BACKEND_BINDING_H_

#include "logging.h"
#include "check.h"

#ifdef INSPIRECV_BACKEND_OPENCV
#define INSPIRECV_BACKEND_TAG "OpenCV"
#include "inspirecv/backends/opencv/adapter/backend_bundle.h"
#else
#define INSPIRECV_BACKEND_TAG "OKCV"
#include "inspirecv/backends/okcv/adapter/backend_bundle.h"
#endif

#endif  // INSPIRECV_CORE_API_BACKEND_BINDING_H_
