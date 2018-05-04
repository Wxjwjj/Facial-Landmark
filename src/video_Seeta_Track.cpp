#include <cstdint>
#include <fstream>
#include <string>
#include "face_detection.h"

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

int main(int argc, char **argv)
{
    // List of tracker types in OpenCV 3.2+
    // NOTE : GOTURN implementation is buggy and does not work.
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN"};
    // vector <string> trackerTypes(types, std::end(types));
 
    // Create a tracker
    string trackerType = trackerTypes[2];
 
    Ptr<Tracker> tracker;
 
    #if (CV_MINOR_VERSION < 3)
    {
        tracker = Tracker::create(trackerType);
    }
    #else
    {
        if (trackerType == "BOOSTING")
            tracker = TrackerBoosting::create();
        if (trackerType == "MIL")
            tracker = TrackerMIL::create();
        if (trackerType == "KCF")
            tracker = TrackerKCF::create();
        if (trackerType == "TLD")
            tracker = TrackerTLD::create();
        if (trackerType == "MEDIANFLOW")
            tracker = TrackerMedianFlow::create();
        if (trackerType == "GOTURN")
            tracker = TrackerGOTURN::create();
    }
    #endif
    // Read video
    VideoCapture video("../videos/chaplin.mp4");
     
    // Exit if video is not opened
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
         
    }
     
    // Read first frame
    Mat frame;
    bool ok = video.read(frame);
    cout<<frame.cols<<" "<<frame.rows<<endl;
     
    Rect2d bbox(0, 0, 0, 0);

    tracker->init(frame, bbox);

    dlib::shape_predictor pose_model;
    dlib::deserialize("../model/dlib_point.dat") >> pose_model;

    seeta::FaceDetection detector_seeta("../model/seeta_fd_frontal_v1.0.bin");

    detector_seeta.SetMinFaceSize(40);
    detector_seeta.SetScoreThresh(2.f);
    detector_seeta.SetImagePyramidScaleFactor(0.8f);
    detector_seeta.SetWindowStep(4, 4);

    Mat img_gray; 
    
    int fflag = 0;
    int flag = 0;
    double timer1 = (double)getTickCount();
    while(video.read(frame))
    {   
        //double timer = (double)getTickCount();
         
        // Update the tracking result
        //bbox坐标为设定的初始值，第一次跟踪会失败，执行下面的face detect
        bool ok = tracker->update(frame, bbox); 
         
        //float fps = getTickFrequency() / ((double)getTickCount() - timer);

        if (flag == 75) 
            ok = false; 

        if (ok)
        {
            flag++;
            // do keypoint
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
            std::vector<dlib::full_object_detection> shapes;
            dlib::cv_image<dlib::bgr_pixel> cimg(frame);
            dlib::rectangle dlibrect; 
            dlibrect = dlib::rectangle((long)bbox.tl().x, (long)bbox.tl().y, (long)bbox.br().x - 1, (long)bbox.br().y - 1);
            shapes.push_back(pose_model(cimg, dlibrect));  
            if (!shapes.empty()) {
                int faceNumber = shapes.size();
                for (int j = 0; j < faceNumber; j++)
                {
                    for (int i = 0; i < 68; i++)
                    {
                        cv::circle(frame, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), 2, cv::Scalar(0, 255, 0), -1);
                    }
                    break;
                }
            }
        }
        else
        {
            flag = 0;
            tracker = TrackerKCF::create(); 

            // do detect
            if (frame.channels() != 1)
              cv::cvtColor(frame, img_gray, cv::COLOR_BGR2GRAY);
            else
              img_gray = frame;

            seeta::ImageData img_data;
            img_data.data = img_gray.data;
            img_data.width = img_gray.cols;
            img_data.height = img_gray.rows;
            img_data.num_channels = 1;

            long t0 = cv::getTickCount();
            std::vector<seeta::FaceInfo> faces = detector_seeta.Detect(img_data);
            long t1 = cv::getTickCount();
            double secs = (t1 - t0)/cv::getTickFrequency();

            int32_t num_face = static_cast<int32_t>(faces.size());

            for (int32_t i = 0; i < num_face; i++) {
            // 判定图像是否超出边界
              if(faces[0].bbox.x<0)
                bbox.x=0;
              else
                  bbox.x = faces[0].bbox.x;
              if(faces[0].bbox.y<0)
                bbox.y=0;
              else
                  bbox.y = faces[0].bbox.y;
              if(bbox.x+faces[0].bbox.width>img_data.width)
                bbox.width=img_data.width-bbox.x;
              else
                  bbox.width = faces[0].bbox.width;
              if(bbox.y+faces[0].bbox.height>img_data.height)
                bbox.height=img_data.height-bbox.y;
              else
                  bbox.height = faces[0].bbox.height;
              bbox.height=bbox.width=MIN(bbox.height, bbox.width);
              break;
            }

            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
            tracker->init(frame, bbox);
        }
        
        // Display frame.
        imshow("Tracking", frame);
         
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }

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
