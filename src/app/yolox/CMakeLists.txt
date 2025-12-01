add_library(app_includes INTERFACE)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Set CUDA architecture for Orin (sm_87)
set(CMAKE_CUDA_ARCHITECTURES 87)

# Find CUDA libraries explicitly
find_library(CUDA_RUNTIME_LIBRARY cudart
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 targets/aarch64-linux/lib)

if(NOT CUDA_RUNTIME_LIBRARY)
    message(FATAL_ERROR "CUDA runtime library not found")
endif()

message(STATUS "Found CUDA runtime: ${CUDA_RUNTIME_LIBRARY}")

# Find TensorRT
set(TensorRT_DIR "/workspace/cgc/TensorRT" CACHE PATH "TensorRT installation directory")
find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS ${TensorRT_DIR}
    PATH_SUFFIXES include)
find_library(TensorRT_LIBRARY nvinfer
    HINTS ${TensorRT_DIR}
    PATH_SUFFIXES lib lib64)

if(NOT TensorRT_INCLUDE_DIR OR NOT TensorRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Please set TensorRT_DIR to the TensorRT installation directory.")
endif()

message(STATUS "Found TensorRT: ${TensorRT_LIBRARY}")
message(STATUS "TensorRT include: ${TensorRT_INCLUDE_DIR}")

target_include_directories(app_includes INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/sink
    ${CMAKE_CURRENT_SOURCE_DIR}/tasks
    ${CMAKE_CURRENT_SOURCE_DIR}/package
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/package 
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIR}
)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/sink/write_consumer APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/source/producer APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/image_preprocess APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/trt_runner APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/object_detector APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/res_sender APP_SRC)

add_executable(yolox_stream yolox_stream.cpp ${SRC_DIR} ${APP_SRC})

target_link_libraries(yolox_stream 
    ${app_includes} 
    ${dynamic_libs} 
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES}
    ${CUDA_RUNTIME_LIBRARY}
    ${TensorRT_LIBRARY}
)

install(TARGETS yolox_stream RUNTIME DESTINATION ./)
