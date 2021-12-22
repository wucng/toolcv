cmake_minimum_required(VERSION 2.6)

project(demo)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace) # 执行程序输出路径


# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")

# 如果你的opencv找不到，可以自己指定目录
# set(OpenCV_DIR   "/data/datav/expstation/lean/opencv4.2.0/lib/cmake/opencv4/")

set(CUDA_DIR     "/usr/local/cuda-10.2")
# set(CUDNN_DIR    "/usr/local/cuda-10.2")
set(TENSORRT_DIR "/home/testuser/Downloads/TensorRT-7.2.3.4")


find_package(CUDA REQUIRED)
#find_package(OpenCV)
# include_directories(OpenCV_INCLUDE_DIRS)

include_directories(
	${PROJECT_SOURCE_DIR}/include/
	# ${PROJECT_SOURCE_DIR}/plugin/
	)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()

# ----------------------------------------------------添加自定义的plugin实现的目录--------------------------------------------------------
# 有.cu文件的需单独添加
# cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/relu.cu)
# target_link_libraries(myplugins nvinfer cudart ${OpenCV_LIBS})
# --------------------------------------------------------------------------------------------------------------------------------------------------------

	
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
#include_directories(/usr/local/cuda/include)
#link_directories(/usr/local/cuda/lib64)

# tensorrt
# include_directories(/usr/include/x86_64-linux-gnu/)
# link_directories(/usr/lib/x86_64-linux-gnu/)

include_directories(
	${CUDA_DIR}/include
	# ${OpenCV_INCLUDE_DIRS}
	${TENSORRT_DIR}/include
	# ${CUDNN_DIR}/include
	)
link_directories(
    ${TENSORRT_DIR}/lib
    ${CUDA_DIR}/lib64
    # ${CUDNN_DIR}/lib
)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
# file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/*.cpp)
# file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/*.cu)
# cuda_add_library(plugin_list SHARED ${cuda_srcs})
# set(SRC_DIR ${PROJECT_SOURCE_DIR}/alex.cpp)
file(GLOB SRC_DIR ${PROJECT_SOURCE_DIR}/*.cpp)

add_executable(demo ${SRC_DIR})
target_link_libraries(demo nvinfer)
target_link_libraries(demo cudart)
target_link_libraries(demo nvonnxparser)
target_link_libraries(demo nvinfer nvinfer_plugin)
#target_link_libraries(demo cuda cublas cudart cudnn)
#target_link_libraries(demo protobuf pthread plugin_list)
#target_link_libraries(demo ${OpenCV_LIBS})
# target_link_libraries(demo myplugins)

add_definitions(-O2 -pthread)

