#include <cstdint>
#include <fstream>
#include <string>
#include "face_detection.h"

#include <dlib/opencv.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <typeinfo>

using namespace dlib;
using namespace std;


int main(int argc, char** argv) {
  if (argc < 3) 
  {
      cout << "Usage: " << argv[0]
          << " image_path model_path"
          << endl;
      return -1;
  }

  const char* img_path = argv[1];
  seeta::FaceDetection detector_seeta(argv[2]);

  detector_seeta.SetMinFaceSize(40);
  detector_seeta.SetScoreThresh(2.f);
  detector_seeta.SetImagePyramidScaleFactor(0.8f);
  detector_seeta.SetWindowStep(4, 4);

  cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
  cv::Mat img_1 = img.clone();
  cv::Mat img_gray;

  if (img.channels() != 1)
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
  else
    img_gray = img;

  seeta::ImageData img_data;
  img_data.data = img_gray.data;
  img_data.width = img_gray.cols;
  img_data.height = img_gray.rows;
  img_data.num_channels = 1;

  long t0 = cv::getTickCount();
  std::vector<seeta::FaceInfo> faces = detector_seeta.Detect(img_data);
  long t1 = cv::getTickCount();
  double secs = (t1 - t0)/cv::getTickFrequency();
  cout << "人脸检测耗时 " << secs << " seconds " << endl;

  cv::Rect face_rect;
  int32_t num_face = static_cast<int32_t>(faces.size());
  ofstream Save("result.txt");
  Save << num_face << endl;

  shape_predictor pose_model; //实例化 key point 检测模型
  deserialize("../model/dlib_point.dat") >> pose_model;

  array2d<rgb_pixel> cimg;
  load_image(cimg, img_path); //dlib方式读取图像（从图像路径）
  
  std::vector<full_object_detection> shapes;

  t0 = cv::getTickCount();
  for (int32_t i = 0; i < num_face; i++) {
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

    Save << "左上角坐标："<< face_rect.x << " " << face_rect.y << " " << "人脸方框宽/高：" << face_rect.width << "\n";

    dlib::rectangle dlibrect; //cv Rect 转 dlib rect
    dlibrect = dlib::rectangle((long)face_rect.tl().x, (long)face_rect.tl().y, (long)face_rect.br().x - 1, (long)face_rect.br().y - 1);
    shapes.push_back(pose_model(cimg, dlibrect));  //调用keypoint检测模型，结果存入shapes。参数1：dlib_image；参数2:dlib_Rect
    cv::rectangle(img_1, face_rect, CV_RGB(255, 255, 255), 1, 8, 0);
  }

  t1 = cv::getTickCount();
  double secss = (t1 - t0)/cv::getTickFrequency();
  cout<<"关键点检测共耗时:"<<secss<<" "<<"秒"<<endl;

  Save.close();
  if (!shapes.empty()) {
    int faceNumber = shapes.size();
    for (int j = 0; j < faceNumber; j++)
      for (int i = 0; i < 68; i++)
      {
          cv::circle(img_1, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 2, cv::Scalar(0, 255, 0), -1);
      }
  } 

  cv::imwrite("Test_box.jpg", img_1);
}
