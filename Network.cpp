#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <Windows.h>
#include <thread>
#include <chrono>
#include "NetworkClass.h"

using namespace dlib;
using namespace std;
using namespace cv;

/**
 * @brief главная функция
 * @return вовращает 0
*/
int main() {
    NetworkClass network;
    network.run();
    return 0;
}