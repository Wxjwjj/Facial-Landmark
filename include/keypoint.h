#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "face_detection.h"

//#define MIN(a,b) ((a)<(b))?(a):(b)
using namespace std;


int detect_point(string path, string path1);
int detect_point(char* src, int srcLength,char**dst);
