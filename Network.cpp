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

std::mutex mtx;
std::condition_variable ConVar;

bool takenSnap = false;
bool stopThreads = false;

void processVideo(cv::VideoCapture& cap) {
    Mat frame; // создания объекта с именем frame. Данный объект представляет собой  структуру данных, 
    //используемую для хранения изображений.
    namedWindow("Video", WINDOW_AUTOSIZE); /*
    Функция namedWindow() создает окно с заданным именем и задает его тип и размер
    (WINDOW_AUTOSIZE указывает, чтобы окно автоматически подстроилось под размер содержимого).
    */
    while (!stopThreads) {
        cap >> frame; // получить новую рамку из камеры
        if (frame.empty()) {
            cout << "Нет рамки!" << endl;
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
            cout << "Изображение сохранено. " << filename << endl;
            takenSnap = true;
            ConVar.notify_one();
            //break;
        }
    }
    //waitKey(0); //Ожидает нажатия клавиши на клавиатуре.
    //destroyAllWindows(); // Закрывает все окна, которые были открыты во время выполнения программы.
}

void threadDescriptor(std::vector<matrix<rgb_pixel>> &faces1, shape_predictor &sp,
    frontal_face_detector &detector, matrix<rgb_pixel> &img1) {
    
    for (auto face : detector(img1))
    {
        
        auto shape = sp(img1, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img1, get_face_chip_details(shape, 150), face_chip);
        faces1.push_back(move(face_chip));
    }
}

int comPort(int res) {
    // Открываем последовательный порт
    HANDLE serialHandle;
    /*
    объявляется переменная `serialHandle` типа `HANDLE`
    `HANDLE` - это дескриптор, используемый для обращения к открытым файлам,
    дескрипторам и различным ресурсам операционной системы Windows
    */

    serialHandle = CreateFile(L"COM3", GENERIC_READ | GENERIC_WRITE, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);/*
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
        //return 1;
    }
    file << res;
    std::string value = to_string(res);
    //if (std::getline(file, value)) {
    std::cout << "Прочитанное значение: " << value << '\n';

    // Отправляем значение в COM-порт
    DWORD bytesWritten;
    if (WriteFile(serialHandle, value.c_str(), value.length(), &bytesWritten, NULL)) {
        std::cout << "Значение успешно отправлено в COM-порт\n";
    }
    //}
    return res;
}

void processNeuralNetwork() {
    while (!stopThreads) {
        auto begin = chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mtx);
        ConVar.wait(lock, [] {return takenSnap; });

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
                auto end = chrono::steady_clock::now();
                auto timeMs = chrono::duration_cast<chrono::milliseconds>(end - begin);
                cout << "Time: " << timeMs.count() << "ms\n";
            }
        }
    }
    std::unique_lock<std::mutex> unlock(mtx);
}

int main()
{
    try {
        setlocale(LC_ALL, "Russian");// Руссификация консоли

        VideoCapture cap(0); // Включаем веб-камеру
        // processVideo(cap);
        // processNeuralNetwork();
        if (!cap.isOpened()) { // Проверка подключении камеры
            cout << " Нет подключения к камере! " << endl; // Вывод в консоль сообщения
            return -1;
        }
        

        thread videoThread(processVideo, ref(cap));
        thread neuralNetworkThread(processNeuralNetwork);

        videoThread.join();
        neuralNetworkThread.join();

        
    }

    catch (...) {
        cout << "Лицо не найдено!" << endl;
    }

    return 0;
}