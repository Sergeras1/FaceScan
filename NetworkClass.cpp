#include "NetworkClass.h"
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
/**
 * @brief ��������� ����� ��� �������� �����������
 * @param cap ������ ���������� � �������� ���������, ����� ����� ��� ������������ ���� ������ ��� ������� ����� �� ���������� ���������
*/
void NetworkClass::processVideo(cv::VideoCapture& cap) {
    Mat frame; // �������� ������� � ������ frame. ������ ������ ������������ �����  ��������� ������, 
    //������������ ��� �������� �����������.
    namedWindow("Video", WINDOW_AUTOSIZE); /*
    ������� namedWindow() ������� ���� � �������� ������ � ������ ��� ��� � ������
    (WINDOW_AUTOSIZE ���������, ����� ���� ������������� ������������ ��� ������ �����������).
    */
    while (!stopThreads) {
        cap >> frame; // �������� ����� ����� �� ������
        if (frame.empty()) {
            cout << "��� �����!\n";
            break;
        }
        imshow("Video", frame); // ���������� ����� �� ����

        char key = (char)waitKey(30); // �������� ������� ������

        if (key == 'c') {

            lock_guard<std::mutex> lock(mtx);
            string filename = "snapshot.jpg";
            imwrite(filename, frame); // ����������� ����������� � ����

            cout << " \n";
            cout << "====================================== \n";
            cout << "����������� ���������. " << filename << "\n";
            takenSnap = true;
            ConVar.notify_one();
        }
    }
}
/**
 * @brief ����������� ��� ������������� �������� ������ ��� �� ��������
*/
NetworkClass::NetworkClass() : takenSnap(false), stopThreads(false){}
/**
 * @brief ����� ��� ���������� ������������ ��� � ����������
 * @param faces1 ������, � ������� ����������� ������� ��� �����������
 * @param sp ������, ������� ������������ ��� ������������ ����� ����
 * @param detector ������, ������� ������������ ��� ����������� �������� ��� �� �����������
 * @param img1 ��� ������� ����������� ����, ������� ������������ ����� �����������, � ������� �������� �������
*/
void NetworkClass::threadDescriptor(std::vector<matrix<rgb_pixel>>& faces1, shape_predictor& sp,
									frontal_face_detector& detector, matrix<rgb_pixel>& img1) {

    for (auto face : detector(img1))
    {
        auto shape = sp(img1, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img1, get_face_chip_details(shape, 150), face_chip);
        faces1.push_back(move(face_chip));
    }
}
/**
 * @brief �����, ������� ������������ ��������� � ������������ ��� � �����������
*/
void NetworkClass::processNeuralNetwork() {
    while (!stopThreads) {
        auto begin = chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mtx);
        ConVar.wait(lock, [this] {return takenSnap; });

        takenSnap = false;
        // ������� ������� ��������� ��� � ������������� ���
        frontal_face_detector detector = get_frontal_face_detector();/*
        ������� ������ `frontal_face_detector` (����������� ����) � �������������� ��� �
        ������� ������� `get_frontal_face_detector()`.
        */
        shape_predictor sp; //������������ ����� ����
        deserialize("C:/Users/sergey/Documents/new_papka/��������� ���������/dat/shape_predictor_68_face_landmarks.dat") >> sp;/*
        ��������� �������������� ������ shape_predictor, �������� ������ �� �����
        "shape_predictor_68_face_landmarks.dat" �� ���������� ����.
        */
        anet_type net;/*
         ������� ������ `anet_type` (������������� ���� �� ������ ������������ ���������, ���������� ������� ����).
        */

        deserialize("C:/Users/sergey/Documents/new_papka/��������� ���������/dat/dlib_face_recognition_resnet_model_v1.dat") >> net;

        // �������� ����������� � ������
        string photo1, photo2;
        //cout << " �������� ���� ��������� ����������: " << endl;
       // cin >> photo1;

        cout << " \n";
        cout << "====================================== \n";
        cout << "��������� �����������... " << endl;
        //cin >> photo2;

        photo2 = "C:/Users/sergey/Documents/new_papka/Visual_Studio_Projects/C++/Network/Network/snapshot.jpg";
        matrix<rgb_pixel> img1, img2;
        load_image(img1, "C:/Users/sergey/Documents/new_papka/��������� ���������/test_image/img1.jpg");
        load_image(img2, photo2);

        // ������� ������� ��� � ��������� ����������� ��� ������� ����
        std::vector<matrix<rgb_pixel>> faces1, faces2;
        threadDescriptor(faces1, sp, detector, img1);
        threadDescriptor(faces2, sp, detector, img2);

        // ���������� ����������� ���
        std::vector<matrix<float, 0, 1>> face_descriptors1 = net(faces1);
        std::vector<matrix<float, 0, 1>> face_descriptors2 = net(faces2);

        // ��������� ���������� ����� �������������
        for (const auto& face_descriptor1 : face_descriptors1)
        {
            for (const auto& face_descriptor2 : face_descriptors2)
            {
                double distance = length(face_descriptor1 - face_descriptor2);
                cout << " \n";
                cout << "====================================== \n";
                cout << " ���������� ����� ������: " << distance << endl;
                int res;

                if (distance > 0.6) {
                    cout << "\n";
                    cout << " �� ���������� ����" << endl;
                    cout << " \n";
                    cout << "====================================== \n";
                    res = 0;
                    comPort(res);
                    /*auto end = chrono::steady_clock::now();
                    auto timeMs = chrono::duration_cast<chrono::milliseconds>(end - begin);
                    cout << "Time: " << timeMs.count() << "ms\n";*/
                }
                else {
                    cout << " \n";
                    cout << "====================================== \n";
                    cout << " ���� ����������" << endl;
                    cout << " \n";
                    cout << "====================================== \n";
                    res = 1;

                    comPort(res);

                }

            }
        }
        auto end = chrono::steady_clock::now();
        auto timeMs = chrono::duration_cast<chrono::milliseconds>(end - begin);
        cout << "Time: " << timeMs.count() << "ms\n";
    }
    std::unique_lock<std::mutex> unlock(mtx);
}
/**
 * @brief ����� ��� ������ ����������� ��������� 
*/
void NetworkClass::run() {
    try {
        setlocale(LC_ALL, "Russian");// ������������ �������

        VideoCapture cap(0); // �������� ���-������
       
        if (!cap.isOpened()) { // �������� ����������� ������
            cout << " ��� ����������� � ������! \n"; // ����� � ������� ���������
        }

        thread videoThread(&NetworkClass::processVideo, this, ref(cap)); 
        thread neuralNetworkThread(&NetworkClass::processNeuralNetwork, this);

        videoThread.join();
        neuralNetworkThread.join();

    }
    catch (...) {
        cout << "���� �� �������!" << endl;
    }
}
/**
 * @brief �����, ���������� �� �������� �������� ���������� res ����� ���������������� ���� COM3. 
 * @param res ����������, ���� ������������ ������������� ��������� �������� ����������
 * @return ���������� ���������
*/
int NetworkClass::comPort(int res) {
    // ��������� ���������������� ����
    HANDLE serialHandle;
    /*
    ����������� ���������� `serialHandle` ���� `HANDLE`
    `HANDLE` - ��� ����������, ������������ ��� ��������� � �������� ������,
    ������������ � ��������� �������� ������������ ������� Windows
    */

    serialHandle = CreateFile(L"COM3", GENERIC_READ | GENERIC_WRITE, 0,
                                0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);/*
    ��������� ���������������� ���� `COM3` ��� ������ � ������.
    `CreateFile` - ��� ������� Windows ��� �������� ����� ��� ����������
    */

    //����������� ������� ��������� ��� COM-�����
    DCB serialParams = { 0 };
    serialParams.DCBlength = sizeof(serialParams); // ������ ��������� `DCB`

    GetCommState(serialHandle, &serialParams);//��������� ������ ����� 
    //����� ���������� ������
    serialParams.BaudRate = CBR_9600;// �������� �������� ������
    serialParams.ByteSize = 8;// ������ �����
    serialParams.StopBits = ONESTOPBIT; // ����-����
    serialParams.Parity = NOPARITY;// �������� ��������
    SetCommState(serialHandle, &serialParams);// ��� ���������� ���� �������� ����������� 

    // Set timeouts
    COMMTIMEOUTS timeout = { 0 };

    /*
    ��������������� �������� ��������� ������ � ������ �
    50 ����������� � 10 �����������, ��������������
    */

    timeout.ReadIntervalTimeout = 50;
    timeout.ReadTotalTimeoutConstant = 50;
    timeout.ReadTotalTimeoutMultiplier = 50;
    timeout.WriteTotalTimeoutConstant = 50;
    timeout.WriteTotalTimeoutMultiplier = 10;

    SetCommTimeouts(serialHandle, &timeout);
    std::ofstream file("data.txt");
    if (!file.is_open()) {
        std::cout << "�� ������� ������� ����\n";
        CloseHandle(serialHandle);
    }
    file << res;
    std::string value = to_string(res);
    std::cout << "����������� ��������: " << value << '\n';

    // ���������� �������� � COM-����
    DWORD bytesWritten;
    if (WriteFile(serialHandle, value.c_str(), value.length(), &bytesWritten, NULL)) {
        std::cout << "�������� ������� ���������� � COM-����\n";
    }
    return res;
}
