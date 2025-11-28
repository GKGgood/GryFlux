add_library(fusionnetv2_app_includes INTERFACE)

find_package(OpenCV REQUIRED)

set(FUSION_APP_DIR ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(fusionnetv2_app_includes INTERFACE
    ${FUSION_APP_DIR}/source
    ${FUSION_APP_DIR}/sink
    ${FUSION_APP_DIR}/tasks
    ${FUSION_APP_DIR}/package
    ${FUSION_APP_DIR}/runtime
)

set(rknn_lib /usr/lib/librknnrt.so)
include_directories(${FUSION_APP_DIR}/package ${FUSION_APP_DIR}/runtime ${OpenCV_INCLUDE_DIRS})

set(APP_SRC)
aux_source_directory(${FUSION_APP_DIR}/sink/write_consumer APP_SRC)
aux_source_directory(${FUSION_APP_DIR}/source/producer APP_SRC)
aux_source_directory(${FUSION_APP_DIR}/tasks/image_preprocess APP_SRC)
aux_source_directory(${FUSION_APP_DIR}/tasks/rk_runner APP_SRC)
aux_source_directory(${FUSION_APP_DIR}/tasks/fusion_composer APP_SRC)
aux_source_directory(${FUSION_APP_DIR}/tasks/res_sender APP_SRC)
aux_source_directory(${FUSION_APP_DIR}/runtime APP_SRC)

add_executable(fusionnetv2_stream fusionnetv2_stream.cpp ${SRC_DIR} ${APP_SRC})

target_link_libraries(fusionnetv2_stream fusionnetv2_app_includes ${dynamic_libs} ${OpenCV_LIBS} ${rknn_lib})

install(TARGETS fusionnetv2_stream RUNTIME DESTINATION ./)
