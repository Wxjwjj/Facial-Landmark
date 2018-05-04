
// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/
#include <cstdint>
#include <fstream>
#include <string>
#include "face_detection.h"

#include <dlib/opencv.h>
#include <opencv2/core/version.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv/cv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <typeinfo>

using namespace dlib;
using namespace std;
using namespace cv;

int main()
{
    try
    {
        cv::VideoCapture cap(0); //打开设备默认摄像头
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        
        cap.set(CV_CAP_PROP_FRAME_WIDTH, 160);  
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 120);  
        
        image_window win; //打开 X11 window

        // Load face detection and pose estimation models.
        //frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("../model/dlib_point.dat") >> pose_model;

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            // Grab a frame
            long t0 = cv::getTickCount();
            cv::Mat temp;
            if (!cap.read(temp)) 
            {
                break;
            }

            seeta::FaceDetection detector_seeta("../model/seeta_fd_frontal_v1.0.bin");

            detector_seeta.SetMinFaceSize(40);
            detector_seeta.SetScoreThresh(2.f);
            detector_seeta.SetImagePyramidScaleFactor(0.8f);
            detector_seeta.SetWindowStep(4, 4);

            cv::Mat img_gray;

            if (temp.channels() != 1)
              cv::cvtColor(temp, img_gray, cv::COLOR_BGR2GRAY);
            else
              img_gray = temp;

            seeta::ImageData img_data;
            img_data.data = img_gray.data;
            img_data.width = img_gray.cols;
            img_data.height = img_gray.rows;
            img_data.num_channels = 1;

          
            std::vector<seeta::FaceInfo> faces = detector_seeta.Detect(img_data);

            cv_image<bgr_pixel> cimg(temp);
            
            cv::Rect face_rect;
            int32_t num_face = static_cast<int32_t>(faces.size());

            std::vector<full_object_detection> shapes;

              for (int32_t i = 0; i < num_face; i++) {  
                  //判定矩形框是否越界
                  if(faces[i].bbox.x<0)
                    face_rect.x=0;
                  else
                      face_rect.x = faces[i].bbox.x;
                  if(faces[i].bbox.y<0)
                    face_rect.y=0;
                  else
                      face_rect.y = faces[i].bbox.y;
                  if(face_rect.x+faces[i].bbox.width>img_data.width)
                    face_rect.width=img_data.width-face_rect.x;
                  else
                      face_rect.width = faces[i].bbox.width;
                  if(face_rect.y+faces[i].bbox.height>img_data.height)
                    face_rect.height=img_data.height-face_rect.y;
                  else
                      face_rect.height = faces[i].bbox.height;

                  face_rect.height=face_rect.width=MIN(face_rect.height, face_rect.width);

                  dlib::rectangle dlibrect; 
                  dlibrect = dlib::rectangle((long)face_rect.tl().x, (long)face_rect.tl().y, (long)face_rect.br().x - 1, (long)face_rect.br().y - 1);
                  shapes.push_back(pose_model(cimg, dlibrect));  

                  cv::rectangle(temp, face_rect, CV_RGB(255, 255, 255), 1, 8, 0);
             }

            if (!shapes.empty()) {
                int faceNumber = shapes.size();
                for (int j = 0; j < faceNumber; j++)
                {
                    for (int i = 0; i < 68; i++)
                    {
                       //用来画特征值的点
                        cv::circle(temp, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 5, cv::Scalar(0, 0, 255), -1);
                       //显示数字
                       //cv::putText(temp,to_string(i), cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), CV_FONT_HERSHEY_PLAIN,1, cv::Scalar(0, 0, 255));
                    }
                }
            }
            // Display it all on the screen
            win.clear_overlay();
            win.set_image(cimg);

            long t1 = cv::getTickCount();
            double secs = (t1 - t0)/cv::getTickFrequency();
            cout << "Seetaface检测耗时：" << secs << " seconds " <<endl;

            //win.add_overlay(render_face_detections(shapes));
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}


