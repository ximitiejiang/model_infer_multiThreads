#pragma once
#include "paddle_deploy.h"
#include "logger.h" // [suliang] LOGC
#include "thread_pool.h"

// model_infer原生api
extern "C" __declspec(dllexport) PaddleDeploy::Model * InitModel(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type);
extern "C" __declspec(dllexport) PaddleDeploy::Model * InitModel_TRT(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type);
extern "C" __declspec(dllexport) void Seg_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, unsigned char* output);
extern "C" __declspec(dllexport) void Seg_ModelBatchPredict(PaddleDeploy::Model * model, const std::vector<unsigned char*> imgs, int nWidth, int nHeight, int nChannel, std::vector<unsigned char*> output);
extern "C" __declspec(dllexport) void DestructModel(PaddleDeploy::Model * model);


// 模型和线程池二次封装
class __declspec(dllexport) ModelWrapper
{
public:
	// 模型初始化
	void InitModelEnter(const char* model_type, const char* model_dir, int gpu_id, bool use_trt);
	// 推理
	void SegPredictEnter(unsigned char* imageData, int width, int height, int channels, unsigned char* result_map);
	// 释放内存
	void DestructModelEnter();

private:
	ThreadPool* pool = NULL;
	PaddleDeploy::Model* _model = NULL;
	int num_threads = 1;
};

// 二次封装后api
extern "C" __declspec(dllexport) ModelWrapper * ModelObjInit(const char* model_type, const char* model_dir, int gpu_id, bool use_trt);
extern "C" __declspec(dllexport) void ModelObjPredict_Seg(ModelWrapper * segObj, unsigned char* imageData, int width, int height, int channels, unsigned char* resultMap);
extern "C" __declspec(dllexport) void ModelObjDestruct(ModelWrapper * segObj);