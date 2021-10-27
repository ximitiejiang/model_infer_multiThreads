#pragma once
#include "paddle_deploy.h"
#include "logger.h" // [suliang] LOGC
#include "thread_pool.h"

// 多线程默认推理api：基于原生model_infer.cpp修改了形参和返回值，把model作为结果返回
extern "C" __declspec(dllexport) PaddleDeploy::Model * InitModel(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type);
extern "C" __declspec(dllexport) PaddleDeploy::Model * InitModel_TRT(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type);
extern "C" __declspec(dllexport) void Seg_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, unsigned char* output);
extern "C" __declspec(dllexport) void Seg_ModelBatchPredict(PaddleDeploy::Model * model, const std::vector<unsigned char*> imgs, int nWidth, int nHeight, int nChannel, std::vector<unsigned char*> output);
extern "C" __declspec(dllexport) void DestructModel(PaddleDeploy::Model * model);


// 增加二次封装类：把模型和线程池封装
class __declspec(dllexport) ModelWrapper
{
public:
	// 模型初始化
	void InitModelEnter(const char* model_type, const char* model_dir, int gpu_id, bool use_trt);
	// 分割模型单图推理
	void SegPredictEnter(unsigned char* imageData, int width, int height, int channels, unsigned char* result_map);
	// 模型资源释放
	void DestructModelEnter();

private:
	// 线程池
	ThreadPool* pool = NULL;
	// 模型
	PaddleDeploy::Model* _model = NULL;
	// 线程池中线程个数
	int num_threads = 1;
};

// 增加多线程二次封装接口
extern "C" __declspec(dllexport) ModelWrapper * ModelObjInit(const char* model_type, const char* model_dir, int gpu_id, bool use_trt);
extern "C" __declspec(dllexport) void ModelObjPredict_Seg(ModelWrapper * segObj, unsigned char * imageData, int width, int height, int channels, unsigned char* resultMap);
extern "C" __declspec(dllexport) void ModelObjDestruct(ModelWrapper * segObj);
