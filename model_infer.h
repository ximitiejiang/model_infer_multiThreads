#pragma once
#include "paddle_deploy.h"
#include "logger.h" // [suliang] LOGC
#include "thread_pool.h"

// 多线程默认推理api：基于原生model_infer.cpp修改了形参和返回值，把model作为结果返回
extern "C" __declspec(dllexport) PaddleDeploy::Model * InitModel(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type);
extern "C" __declspec(dllexport) PaddleDeploy::Model * InitModel_TRT(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type,
	std::vector<int>min_input_shape, std::vector<int>max_input_shape, std::vector<int>optim_input_shape, int precision=0, int min_subgraph_size=40);
extern "C" __declspec(dllexport) void DestructModel(PaddleDeploy::Model * model);
// 分割
extern "C" __declspec(dllexport) void Seg_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, unsigned char* output);
extern "C" __declspec(dllexport) void Seg_ModelBatchPredict(PaddleDeploy::Model * model, const std::vector<unsigned char*> imgs, int nWidth, int nHeight, int nChannel, std::vector<unsigned char*> output);
// 检测
extern "C" __declspec(dllexport) void Det_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* output, int* nBoxesNum, char* LabelList);
// 分类
extern "C" __declspec(dllexport) void Cls_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* score, char* category, int* category_id);
// Mask
extern "C" __declspec(dllexport) void Mask_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList);


// 增加二次封装类：把模型和线程池封装
class __declspec(dllexport) ModelWrapper
{
public:
	// 模型初始化
	void InitModelEnter(const char* model_type, const char* model_dir, int gpu_id, bool use_trt, const std::vector<int>min_input_shape = std::vector<int>(), const std::vector<int>max_input_shape = std::vector<int>(), const std::vector<int>optim_input_shape = std::vector<int>(), int precision = 0, int min_subgraph_size = 40);
	// 分割模型单图推理
	void SegPredictEnter(unsigned char* imageData, int width, int height, int channels, unsigned char* result_map);
	// 检测模型
	void DetPredictEnter(unsigned char* imageData, int width, int height, int channels, float* output, int* nBoxesNum, char* LabelList);
	// 分类模型
	void ClsPredictEnter(unsigned char* imageData, int width, int height, int channels, float* score, char* category, int* category_id);
	// Mask模型
	void MaskPredictEnter(unsigned char* imageData, int width, int height, int channels, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList);
	// 模型资源释放
	void DestructModelEnter();

private:
	ThreadPool* pool = NULL;
	PaddleDeploy::Model* _model = NULL;
	int num_threads = 1;
};

// 增加多线程二次封装接口
extern "C" __declspec(dllexport) ModelWrapper * ModelObjInit(const char* model_type, const char* model_dir, int gpu_id, bool use_trt, 
	const std::vector<int>min_input_shape=std::vector<int>(), const std::vector<int>max_input_shape = std::vector<int>(), const std::vector<int>optim_input_shape = std::vector<int>(), int precision=0, int min_subgraph_size=40);
extern "C" __declspec(dllexport) void ModelObjDestruct(ModelWrapper * modelObj);
extern "C" __declspec(dllexport) void ModelObjPredict_Seg(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, unsigned char* resultMap);
extern "C" __declspec(dllexport) void ModelObjPredict_Det(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* output, int* nBoxesNum, char* LabelList);
extern "C" __declspec(dllexport) void ModelObjPredict_Cls(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* score, char* category, int* category_id);
extern "C" __declspec(dllexport) void ModelObjPredict_Mask(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList);
