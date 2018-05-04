cmake_minimum_required(VERSION 2.8.12)

project(examples)

add_subdirectory(dlib dlib_build)  

add_subdirectory(FaceDetection Face_build)

include_directories(${PROJECT_SOURCE_DIR}/include)

# 若使用自己安装的OpenCV，执行如下指令
find_package(OpenCV)
if (NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found. Test will not be built.")
else()
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# 若使用笔者在Mac下编译的opencv库，请见CMakeLists_Mac_CV.txt

add_executable(video_Seeta_Track src/video_Seeta_Track.cpp)
target_link_libraries(video_Seeta_Track dlib seeta_facedet_lib ${OpenCV_LIBS})

add_executable(real_dlib_muban_roi src/real_dlib_muban_roi.cpp)
target_link_libraries(real_dlib_muban_roi dlib seeta_facedet_lib ${OpenCV_LIBS})

add_executable(real_dlib src/real_dlib.cpp)
target_link_libraries(real_dlib dlib seeta_facedet_lib ${OpenCV_LIBS})

add_executable(real_seeta src/real_seeta.cpp)
target_link_libraries(real_seeta dlib seeta_facedet_lib ${OpenCV_LIBS})

add_executable(Pic_LandMark68 src/Pic_LandMark68.cpp)
target_link_libraries(Pic_LandMark68 dlib seeta_facedet_lib ${OpenCV_LIBS})


