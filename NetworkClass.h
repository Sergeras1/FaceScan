#pragma once
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

using namespace dlib;
using namespace std;
using namespace cv;

//Архитектура сверторочной нейросети для обработки изображения, основанная на блоках residual и residual_down. 

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>

using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>; /* Сверторочный блок с функцией активации relu и слоя
add_prev1, который добавляет остаток к предыдущему выходу.
*/

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>

using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;/*
residual_down - аналогичен residual, но использует слой avg_pool для уменьшения размерности входных данных перед сверточным слоем.
*/

template <int N, template <typename> class BN, int stride, typename SUBNET>

using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;/*
block - представляет собой базовый блок с сверточным слоем, функцией активации relu и слоем BN (batch normalization).
*/
template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;/*
ares - блок, состоящий из нескольких последовательных residual блоков.
*/

template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;

template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;

template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;

template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;

template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric < fc_no_bias < 128, avg_pool_everything< /*
    - alevel0, alevel1, alevel2, alevel3, alevel4 - последовательность блоков ares и
 residual_down с разными параметрами для создания иерархии архитектуры нейронной сети.

    - anet_type - основной тип нейронной сети, который использует все определенные выше блоки
и слои для создания итоговой архитектуры. Она используется для обучения с использованием
loss_metric и включает в себя последовательность сверточных слоев, функций активации,
пулинга и других операций для обработки изображений размером 150x150 с использованием
архитектуры представленных блоков. В конце есть слой fc_no_bias для получения
128-мерного признакового вектора изображения.
    */

    alevel0<

    alevel1<

    alevel2<

    alevel3<

    alevel4<

    max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,

    input_rgb_image_sized<150>

    >>>>>>>>>>>>;
/**
 * @brief класс нейросети
*/
class NetworkClass {
private:
    /**
     * @brief Объявление переменных
    */
    std::mutex mtx;
    std::condition_variable ConVar;
    bool takenSnap;
    bool stopThreads;

public:
    /**
     * @brief объявление конкструктора и методов класса
    */
    NetworkClass();
    void processVideo(cv::VideoCapture& cap);
    void threadDescriptor(std::vector<matrix<rgb_pixel>>& faces1, shape_predictor& sp,
                          frontal_face_detector& detector, matrix<rgb_pixel>& img1);
    void processNeuralNetwork();
    void run();
    int comPort(int res);
};

