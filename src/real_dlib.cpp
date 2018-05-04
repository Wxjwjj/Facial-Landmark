#include <dlib/opencv.h>
#include <opencv2/core/version.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;

int main()
{
    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;

        //摄像头分别率设置，若要提升分辨率，建议不要采用resize函数，这样会增加耗时。
        //可增加两路视频信号输入，低分辨率计算关键点，将坐标进行相应变换，在高分辨率视频中呈现
        
        cap.set(CV_CAP_PROP_FRAME_WIDTH, 160);  
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 120);  

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
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

            cv_image<bgr_pixel> cimg(temp);

            // 人脸检测
            std::vector<rectangle> faces = detector(cimg);

            // 绘制每个脸部矩形框
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i){
                cout<<(long)(faces[i].left())<<endl;
                shapes.push_back(pose_model(cimg, faces[i])); 
                cv::rectangle(temp, cvPoint(faces[i].left(), faces[i].top()),cvPoint(faces[i].right(),faces[i].bottom()), CV_RGB(255, 255, 255), 1, 8, 0);
            }

            // 绘制每个脸部关键点
            if (!shapes.empty()) {
                int faceNumber = shapes.size();
                for (int j = 0; j < faceNumber; j++)
                {
                    for (int i = 0; i < 68; i++)
                    {
                        cv::circle(temp, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 3, cv::Scalar(0, 0, 255), -1);
                    }
                }
            }
            // 播放及耗时计算
            win.clear_overlay();
            win.set_image(cimg);
            long t1 = cv::getTickCount();
            double secss = (t1 - t0)/cv::getTickFrequency();
            cout<<"dlib检测耗时:"<<" "<<secss<<endl;
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

