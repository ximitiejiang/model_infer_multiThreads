# model_infer_multiThreads

该repo基于PaddleX模型推理动态链接库的接口代码进行修改，支持多线程并行访问。大部分代码均来自paddleX的model_infer.cpp
#### 最近更新：
- 2021-11-24 增加demo_cs, demo_cpp工程
- 2021-10-28 增加了原生的所有api接口，支持clas/det/seg/mask)

## 修改部分：
- 把模型作为对象从初始化函数返回，便于进行多线程控制
- 对模型和线程池进行二次封装，避免了多线程调用时推理失败

## 使用方法：
替换原有的model_infer项目中的部分文件，实现model_infer.dll的多线程访问的支持，具体如下:

- 基于paddleX的model_infer动态库代码，确保model_infer.dll能够正确生成，并能够整理依赖的dll，让model_infer.dll在单线程下运行正常
- 把该repo中的model_infer.cpp替代原来model_infer动态库代码中的model_infer.cpp，同时把model_infer.h/thread_pool.h/logger.h都拷贝到model_deploy/common/include文件夹中
![files_position](https://user-images.githubusercontent.com/24242483/139017800-78736d89-f2cc-452b-9d99-82361fa8be6e.png)

- 在model_infer动态库代码项目中右键/属性/C/C++/常规/附件包含目录，添加PaddleX-release-2.0.0\deploy\cpp文件夹，从而保证整个model_deploy文件夹都能被model_infer项目找到，确保可以成功生成model_infer.dll
![add_head_folder](https://user-images.githubusercontent.com/24242483/139017936-44a5399f-c203-4842-9a58-4ff4ffcbfd7f.png)

- 使用该repo的demo_cs代码验证用c#多线程访问模型
该工程下载后，需要添加对opencvsharp的引用，同时还需要把前几步生成好的多线程model_infer.dll，以及其他paddle_inference相关的dll都拷贝到exe目录即可运行
![image](https://user-images.githubusercontent.com/24242483/143397023-f046a327-4446-4f2c-bae2-0d89fb2653d5.png)

- 使用该repo的deomo_cpp代码验证用cpp多线程访问模型
该工程下载后，需要在项目/属性中，分别增加对model_infer, opencv, yaml-cpp这几个库的头文件夹、库文件夹、库文件名的引入，以及model_infer.dll和其他图paddle_inference相关的dll拷贝到exe目录即可运行

![image](https://user-images.githubusercontent.com/24242483/143397092-0bc0f774-92fe-487d-a5c8-3420943c759b.png)

##### 注意
- 模型/配置文件的名称被在代码里边固定了，如果不匹配会报错，如果需要可修改model_infer.cpp代码
- 默认只能在gpu下跑，cpu被在代码里边关掉了，如果需要可修改model_infer.cpp代码

![result](https://user-images.githubusercontent.com/24242483/139020183-f0b997c1-c293-4de9-bb72-e3ca8b9185ef.png)

## 注意：
- paddle的模型不支持多线程并行推理，虽然通过改造后，模型可以被多线程访问，但本质上底层是被线程池锁掉了，每次实际上只有单线程访问模型。如果强行让多线程并行访问模型，必然导致推理报错。
- paddle的模型在多线程分时访问时，也不支持不同线程反复创建、销毁，同样会导致底层cuda报错。
- 该repo中所有原生api接口都已导出，但仅简单验证了clas/det/seg部分的功能，部分代码可能存在一些疏漏和错误，仅作为一个抛砖引玉，大家可在上面自行修改。后续项目稳定后我会尽快把完整测试版本更新上面
