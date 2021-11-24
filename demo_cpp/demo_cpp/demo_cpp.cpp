#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include "timer.h"
#include "model_deploy/common/include/model_infer.h"    // 因为引入了paddle_inference的头文件，所以还要增加yaml-cpp引用

// 因logger.h的引入，还要增加"属性/C/C++/预处理器"里边的_CRT_SECURE_NO_WARNINGS

void TestSingleModelMultiThreads()
{
    const char* model_dir = "D:\\suliang\\My_Lib11\\models\\2";
    int gpu_id = 0;
    bool use_trt = false;
    const char* model_type = "seg";
    ModelWrapper* model = ModelObjInit(model_type, model_dir, gpu_id, use_trt);
    ModelWrapper* model2 = ModelObjInit(model_type, model_dir, gpu_id, use_trt);

    cv::Mat src1 = cv::imread("D:\\suliang\\My_Dataset\\stain1k_test\\31042232357036_down_2k_lb.bmp", cv::IMREAD_GRAYSCALE);
    cv::Mat src2 = cv::imread("D:\\suliang\\My_Dataset\\stain1k_test\\31050024054020_up_2k_lt.bmp", cv::IMREAD_GRAYSCALE);

    int width = src1.cols;
    int height = src1.rows;
    int channels = src1.channels();
    printf("image width=%d, height=%d, channels=%d", width, height, channels);
    unsigned char* result_map1 = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char* result_map2 = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    Timer timer;

    // 启动多线程预热
    timer.start();
    std::thread t1(ModelObjPredict_Seg, model, src1.data, width, height, channels, result_map1);
    std::thread t2(ModelObjPredict_Seg, model2, src2.data, width, height, channels, result_map2);
    t1.join();
    t2.join();
    timer.stop_and_show("one model warmup time:");
    std::cout << "finished warmup" << std::endl;

    // 单模型跑1次的时间
    timer.start();
    std::thread t0(ModelObjPredict_Seg, model, src1.data, width, height, channels, result_map1);
    t0.join();
    timer.stop_and_show("1 model with 1 thread to predict 1 pic:");

    // 单模型跑两次的时间
    timer.start();
    std::thread t3(ModelObjPredict_Seg, model, src1.data, width, height, channels, result_map1);
    std::thread t4(ModelObjPredict_Seg, model, src2.data, width, height, channels, result_map2);
    t3.join();
    t4.join();
    timer.stop_and_show("1 model with 2 thread to predict 2 pics:");

    // 两个模型分别预测单图的时间
    timer.start();
    std::thread t5(ModelObjPredict_Seg, model, src1.data, width, height, channels, result_map1);
    std::thread t6(ModelObjPredict_Seg, model2, src2.data, width, height, channels, result_map2);
    t5.join();
    t6.join();
    timer.stop_and_show("2 model with 2 thread to predict 2 pics:");

    //ModelObjPredict_Seg(model, src1.data, width, height, channels, result_map1);
    cv::Mat dst1 = cv::Mat(height, width, CV_8UC1, result_map1);
    cv::Mat dst2 = cv::Mat(height, width, CV_8UC1, result_map2);
    cv::imwrite("D:\\result_map1.bmp", dst1);
    cv::imwrite("D:\\result_map2.bmp", dst2);

    free(result_map1);
    free(result_map2);
}

int main()
{
    TestSingleModelMultiThreads();
}
