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
 * @brief Добавляем метод для создания видеопотока
 * @param cap Объект передается в качестве параметра, чтобы метод мог использовать этот объект для захвата видео из указанного источника
*/
void NetworkClass::processVideo(cv::VideoCapture& cap) {
    Mat frame; // создания объекта с именем frame. Данный объект представляет собой  структуру данных, 
    //используемую для хранения изображений.
    namedWindow("Video", WINDOW_AUTOSIZE); /*
    Функция namedWindow() создает окно с заданным именем и задает его тип и размер
    (WINDOW_AUTOSIZE указывает, чтобы окно автоматически подстроилось под размер содержимого).
    */
    while (!stopThreads) {
        cap >> frame; // получить новую рамку из камеры
        if (frame.empty()) {
            cout << "Нет рамки!\n";
            break;
        }
        imshow("Video", frame); // показывать рамку из окна

        char key = (char)waitKey(30); // получить нажатие кнопки

        if (key == 'c') {

            lock_guard<std::mutex> lock(mtx);
            string filename = "snapshot.jpg";
            imwrite(filename, frame); // сохраняется изображение в файл

            cout << " \n";
            cout << "====================================== \n";
            cout << "Изображение сохранено. " << filename << "\n";
            takenSnap = true;
            ConVar.notify_one();
        }
    }
}
/**
 * @brief конструктор для инициализации объектов класса при их создании
*/
NetworkClass::NetworkClass() : takenSnap(false), stopThreads(false){}
/**
 * @brief метод для вычисления дескрипторов лиц с фотографий
 * @param faces1 Вектор, в котором добавляются области лиц изображений
 * @param sp Объект, который используется для предсказания формы лицы
 * @param detector Объект, который используется для обнаружения областей лиц на изображении
 * @param img1 Это матрица изображения типа, которая представляет собой изображение, с которым работает функция
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
 * @brief метод, который представляет нейросеть с вычислениями лиц с изображений
*/
void NetworkClass::processNeuralNetwork() {
    while (!stopThreads) {
        auto begin = chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mtx);
        ConVar.wait(lock, [this] {return takenSnap; });

        takenSnap = false;
        // Создаем объекты детектора лиц и распознавания лиц
        frontal_face_detector detector = get_frontal_face_detector();/*
        Создает объект `frontal_face_detector` (обнаружение лица) и инициализирует его с
        помощью функции `get_frontal_face_detector()`.
        */
        shape_predictor sp; //предсказание формы лица
        deserialize("C:/Users/sergey/Documents/new_papka/Наработки нейросети/dat/shape_predictor_68_face_landmarks.dat") >> sp;/*
        Выполняет десериализацию модели shape_predictor, загружая данные из файла
        "shape_predictor_68_face_landmarks.dat" по указанному пути.
        */
        anet_type net;/*
         Создает объект `anet_type` (распознавание лица на основе сверторочной нейросети, указзанной вначале кода).
        */

        deserialize("C:/Users/sergey/Documents/new_papka/Наработки нейросети/dat/dlib_face_recognition_resnet_model_v1.dat") >> net;

        // Загрузка изображений с лицами
        string photo1, photo2;
        //cout << " Вставьте путь эталонной фотографии: " << endl;
       // cin >> photo1;

        cout << " \n";
        cout << "====================================== \n";
        cout << "Обработка изображения... " << endl;
        //cin >> photo2;

        photo2 = "C:/Users/sergey/Documents/new_papka/Visual_Studio_Projects/C++/Network/Network/snapshot.jpg";
        matrix<rgb_pixel> img1, img2;
        load_image(img1, "C:/Users/sergey/Documents/new_papka/Наработки нейросети/test_image/img1.jpg");
        load_image(img2, photo2);

        // Находим области лиц и вычисляем дескрипторы для каждого лица
        std::vector<matrix<rgb_pixel>> faces1, faces2;
        threadDescriptor(faces1, sp, detector, img1);
        threadDescriptor(faces2, sp, detector, img2);

        // Сравниваем дескрипторы лиц
        std::vector<matrix<float, 0, 1>> face_descriptors1 = net(faces1);
        std::vector<matrix<float, 0, 1>> face_descriptors2 = net(faces2);

        // Вычисляем расстояния между дескрипторами
        for (const auto& face_descriptor1 : face_descriptors1)
        {
            for (const auto& face_descriptor2 : face_descriptors2)
            {
                double distance = length(face_descriptor1 - face_descriptor2);
                cout << " \n";
                cout << "====================================== \n";
                cout << " Расстояние между лицами: " << distance << endl;
                int res;

                if (distance > 0.6) {
                    cout << "\n";
                    cout << " Не распознано лицо" << endl;
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
                    cout << " Лицо распознано" << endl;
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
 * @brief метод для вывода результатов нейросети 
*/
void NetworkClass::run() {
    try {
        setlocale(LC_ALL, "Russian");// Руссификация консоли

        VideoCapture cap(0); // Включаем веб-камеру
       
        if (!cap.isOpened()) { // Проверка подключении камеры
            cout << " Нет подключения к камере! \n"; // Вывод в консоль сообщения
        }

        thread videoThread(&NetworkClass::processVideo, this, ref(cap)); 
        thread neuralNetworkThread(&NetworkClass::processNeuralNetwork, this);

        videoThread.join();
        neuralNetworkThread.join();

    }
    catch (...) {
        cout << "Лицо не найдено!" << endl;
    }
}
/**
 * @brief метод, отвечающий за отправку значения переменной res через последовательный порт COM3. 
 * @param res переменная, куда записывается окончательный результат выданной нейросетью
 * @return возвращает результат
*/
int NetworkClass::comPort(int res) {
    // Открываем последовательный порт
    HANDLE serialHandle;
    /*
    объявляется переменная `serialHandle` типа `HANDLE`
    `HANDLE` - это дескриптор, используемый для обращения к открытым файлам,
    дескрипторам и различным ресурсам операционной системы Windows
    */

    serialHandle = CreateFile(L"COM3", GENERIC_READ | GENERIC_WRITE, 0,
                                0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);/*
    открывает последовательный порт `COM3` для чтения и записи.
    `CreateFile` - это функция Windows для открытия файла или устройства
    */

    //Выполняются базовые настройки для COM-порта
    DCB serialParams = { 0 };
    serialParams.DCBlength = sizeof(serialParams); // размер структуры `DCB`

    GetCommState(serialHandle, &serialParams);//получение данных порта 
    //Далее изменяются данные
    serialParams.BaudRate = CBR_9600;// скорость передачи данных
    serialParams.ByteSize = 8;// размер байта
    serialParams.StopBits = ONESTOPBIT; // стоп-биты
    serialParams.Parity = NOPARITY;// проверка четности
    SetCommState(serialHandle, &serialParams);// все измененные выше значения сохраняются 

    // Set timeouts
    COMMTIMEOUTS timeout = { 0 };

    /*
    установливаются значения таймаутов чтения и записи в
    50 миллисекунд и 10 миллисекунд, соответственно
    */

    timeout.ReadIntervalTimeout = 50;
    timeout.ReadTotalTimeoutConstant = 50;
    timeout.ReadTotalTimeoutMultiplier = 50;
    timeout.WriteTotalTimeoutConstant = 50;
    timeout.WriteTotalTimeoutMultiplier = 10;

    SetCommTimeouts(serialHandle, &timeout);
    std::ofstream file("data.txt");
    if (!file.is_open()) {
        std::cout << "Не удалось открыть файл\n";
        CloseHandle(serialHandle);
    }
    file << res;
    std::string value = to_string(res);
    std::cout << "Прочитанное значение: " << value << '\n';

    // Отправляем значение в COM-порт
    DWORD bytesWritten;
    if (WriteFile(serialHandle, value.c_str(), value.length(), &bytesWritten, NULL)) {
        std::cout << "Значение успешно отправлено в COM-порт\n";
    }
    return res;
}
