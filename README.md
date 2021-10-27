# model_infer_multiThreads

该repo基于PaddleX模型推理动态链接库的接口代码进行修改，支持多线程并行访问。大部分代码均来自paddleX的model_infer.cpp
修改部分：
- 把模型作为对象从初始化函数返回，便于进行多线程控制
- 对模型和线程池进行二次封装，避免了多线程调用时推理失败

使用方法：
替换原有的model_infer项目中的部分文件，实现model_infer.dll的多线程访问的支持，具体如下
1. 基于paddleX的model_infer动态库代码，确保model_infer.dll能够正确生成，并能够整理依赖的dll，让model_infer.dll在单线程下运行正常
2. 把该repo中的model_infer.cpp替代原来model_infer动态库代码中的model_infer.cpp，同时把model_infer.h/thread_pool.h/logger.h都拷贝到model_deploy/common/include文件夹中
![files_position](https://user-images.githubusercontent.com/24242483/139017800-78736d89-f2cc-452b-9d99-82361fa8be6e.png)

4. 在model_infer动态库代码项目中右键/属性/C/C++/常规/附件包含目录，添加PaddleX-release-2.0.0\deploy\cpp文件夹，从而保证整个model_deploy文件夹都能被model_infer项目找到，确保可以成功生成model_infer.dll
![add_head_folder](https://user-images.githubusercontent.com/24242483/139017936-44a5399f-c203-4842-9a58-4ff4ffcbfd7f.png)

注意：
- paddle的模型不支持多线程并行推理，虽然通过改造后，模型可以被多线程访问，但本质上底层是被线程池锁掉了，每次实际上只有单线程访问模型。如果强行让多线程并行访问模型，必然导致推理报错。
- paddle的模型在多线程分时访问时，也不支持不同线程反复创建、销毁，同样会导致底层cuda报错。
- 该repo中仅验证了seg部分的功能，其他det/cls的api接口没有导出，也没有验证，如有需要可按照seg部分修改即可
