cmake_minimum_required(VERSION 2.8.12)

project(detect)   # 如果此处未重新定义project，那么${PROJECT_SOURCE_DIR}指的是上一级cmake所指的目录

include_directories(${PROJECT_SOURCE_DIR}/../include)

set(src_files 
    src/util/nms.cpp
    src/util/image_pyramid.cpp
    src/io/lab_boost_model_reader.cpp
    src/io/surf_mlp_model_reader.cpp
    src/feat/lab_feature_map.cpp
    src/feat/surf_feature_map.cpp
    src/classifier/lab_boosted_classifier.cpp
    src/classifier/mlp.cpp
    src/classifier/surf_mlp.cpp
    src/face_detection.cpp
    src/fust.cpp
    )

# Build shared library
add_library(seeta_facedet_lib STATIC ${src_files}) 