#include <cstdint>
#include <fstream>
#include <string>
#include <dlib/opencv.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <typeinfo>

using namespace cv;
using namespace std;
using namespace dlib;


void drawcircle(std::vector<dlib::full_object_detection> shapes, cv::Mat frame, Rect2d bbox)
{
    // 人脸关键点绘制
    if (!shapes.empty()) {
        int faceNumber = shapes.size();
        for (int j = 0; j < faceNumber; j++)
        {
            for (int i = 0; i < 68; i++)
            {
                cv::circle(frame, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), 2, cv::Scalar(0, 255, 0), -1);
            }
            break; // 仅测试检测一张人脸，所以进行break
        }
    }
}

int main(int argc, char **argv)
{
    VideoCapture video(0);

    video.set(CV_CAP_PROP_FRAME_WIDTH, 320);  
    video.set(CV_CAP_PROP_FRAME_HEIGHT, 240);  

    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
         
    }
     
    dlib::image_window win; //打开 X11 window

    Mat frame;
    bool ok = video.read(frame);

    Rect2d bbox(-1, -1, 0, 0);

    frontal_face_detector detector = get_frontal_face_detector();
    dlib::shape_predictor pose_model;
    dlib::deserialize("../dlib_point.dat") >> pose_model;

    int fflag = 0;
    int32_t num_face = 0;
    double timer1 = (double)getTickCount();

    while(video.read(frame))
    {   
        Mat frame1 = frame.clone();

        if (bbox.x < 0 || bbox.y < 0 || bbox.x+bbox.width > frame.cols || bbox.y+bbox.height > frame.rows){
            cout<<"bbox broken..."<<endl;

            cv_image<bgr_pixel> cimg(frame);
            std::vector<dlib::rectangle> faces = detector(cimg);
            std::vector<dlib::full_object_detection> shapes;
            num_face = static_cast<int32_t>(faces.size());
            for (int32_t i = 0; i < num_face; i++) {
                shapes.push_back(pose_model(cimg, faces[i])); 
                bbox.x = faces[i].left();
                bbox.y = faces[i].top();
                bbox.width = faces[i].right()-faces[i].left();
                bbox.height = faces[i].bottom()-faces[i].top();
                cv::rectangle(frame1, bbox, CV_RGB(255, 255, 255), 1, 8, 0 );
                break;
            }
            drawcircle(shapes, frame1, bbox);
        }   // 假如bbox越界(人脸出界），则在整张图中搜索人脸
        else
        {
            // 设定ROI区域，前一次的宽、高均放大120%
            Mat image_matched;
            Mat image_template;
            double minVal;
            double maxVal = 0;
            cv::Point minLoc, maxLoc;
            cv::Mat img_gray;

            if (frame.channels() != 1)
                cv::cvtColor(frame, img_gray, cv::COLOR_BGR2GRAY);
            else
                img_gray = frame;

            image_template = img_gray(bbox); 

            int R1 = bbox.width * 1.2;  
            int R2 = bbox.height * 1.2;
            bbox.x = bbox.x - (R1-bbox.width)/2;
            bbox.y = bbox.y - (R2-bbox.height)/2; 
            bbox.width = R1;
            bbox.height = R2;

            if (bbox.x<0)
                bbox.x = 0;
            if (bbox.y<0)
                bbox.y = 0;
            if (bbox.x+bbox.width > frame.cols)
                bbox.width = frame.cols - bbox.x;
            if (bbox.y+bbox.height > frame.rows)
                bbox.height = frame.rows - bbox.y;

            img_gray = img_gray(bbox); 

            cv::matchTemplate(img_gray, image_template, image_matched, cv::TM_CCOEFF_NORMED); 
            cv::minMaxLoc(image_matched, &minVal, &maxVal, &minLoc, &maxLoc);
            if (maxVal == 1){ 
                // 模板匹配成功
                bbox.x = maxLoc.x + bbox.x;
                bbox.y = maxLoc.y + bbox.y;
                bbox.width = image_template.cols;
                bbox.height = image_template.rows;
                cv::rectangle(frame1, bbox, CV_RGB(255, 255, 255), 1, 8, 0);

                cv_image<bgr_pixel> cimg(frame);
                std::vector<dlib::full_object_detection> shapes;
                dlib::rectangle dlibrect; 
                dlibrect = dlib::rectangle((long)bbox.tl().x, (long)bbox.tl().y, (long)bbox.br().x - 1, (long)bbox.br().y - 1);
                shapes.push_back(pose_model(cimg, dlibrect));  
                drawcircle(shapes, frame1, bbox);
            }
            else{
                // 模板匹配失败，则在整张图中搜索人脸
                cv_image<bgr_pixel> cimg(frame);
                std::vector<dlib::rectangle> faces = detector(cimg);
                std::vector<dlib::full_object_detection> shapes;
                num_face = static_cast<int32_t>(faces.size());
                for (int32_t i = 0; i < num_face; i++) {
                    shapes.push_back(pose_model(cimg, faces[i])); 
                    bbox.x = faces[i].left();
                    bbox.y = faces[i].top();
                    bbox.width = faces[i].right()-faces[i].left();
                    bbox.height = faces[i].bottom()-faces[i].top();
                    cv::rectangle(frame1, bbox, CV_RGB(255, 255, 255), 1, 8, 0 );
                    break;
                }
                drawcircle(shapes, frame1, bbox);
            }
        }

        cv_image<bgr_pixel> cimg1(frame1);
        win.clear_overlay();
        win.set_image(cimg1);

        double tt1 = ((double)getTickCount() - timer1)/getTickFrequency();
        if (tt1 < double(1.0)){
            fflag++;
        }
        else{
            cout<<"实时帧率："<<fflag<<endl;
            fflag = 0;   
            timer1 = (double)getTickCount();         
        }

    }
}

