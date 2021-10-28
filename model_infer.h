#pragma once
#include "paddle_deploy.h"
#include "logger.h" // [suliang] LOGC
#include "thread_pool.h"

// ���߳�Ĭ������api������ԭ��model_infer.cpp�޸����βκͷ���ֵ����model��Ϊ�������
extern "C" __declspec(dllexport) PaddleDeploy::Model * InitModel(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type);
extern "C" __declspec(dllexport) PaddleDeploy::Model * InitModel_TRT(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type);
extern "C" __declspec(dllexport) void DestructModel(PaddleDeploy::Model * model);
// �ָ�
extern "C" __declspec(dllexport) void Seg_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, unsigned char* output);
extern "C" __declspec(dllexport) void Seg_ModelBatchPredict(PaddleDeploy::Model * model, const std::vector<unsigned char*> imgs, int nWidth, int nHeight, int nChannel, std::vector<unsigned char*> output);
// ���
extern "C" __declspec(dllexport) void Det_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* output, int* nBoxesNum, char* LabelList);
// ����
extern "C" __declspec(dllexport) void Cls_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* score, char* category, int* category_id);
// Mask
extern "C" __declspec(dllexport) void Mask_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList);


// ���Ӷ��η�װ�ࣺ��ģ�ͺ��̳߳ط�װ
class __declspec(dllexport) ModelWrapper
{
public:
	// ģ�ͳ�ʼ��
	void InitModelEnter(const char* model_type, const char* model_dir, int gpu_id, bool use_trt);
	// �ָ�ģ�͵�ͼ����
	void SegPredictEnter(unsigned char* imageData, int width, int height, int channels, unsigned char* result_map);
	// ���ģ��
	void DetPredictEnter(unsigned char* imageData, int width, int height, int channels, float* output, int* nBoxesNum, char* LabelList);
	// ����ģ��
	void ClsPredictEnter(unsigned char* imageData, int width, int height, int channels, float* score, char* category, int* category_id);
	// Maskģ��
	void MaskPredictEnter(unsigned char* imageData, int width, int height, int channels, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList);
	// ģ����Դ�ͷ�
	void DestructModelEnter();

private:
	ThreadPool* pool = NULL;
	PaddleDeploy::Model* _model = NULL;
	int num_threads = 1;
};

// ���Ӷ��̶߳��η�װ�ӿ�
extern "C" __declspec(dllexport) ModelWrapper * ModelObjInit(const char* model_type, const char* model_dir, int gpu_id, bool use_trt);
extern "C" __declspec(dllexport) void ModelObjDestruct(ModelWrapper * modelObj);
extern "C" __declspec(dllexport) void ModelObjPredict_Seg(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, unsigned char* resultMap);
extern "C" __declspec(dllexport) void ModelObjPredict_Det(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* output, int* nBoxesNum, char* LabelList);
extern "C" __declspec(dllexport) void ModelObjPredict_Cls(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* score, char* category, int* category_id);
extern "C" __declspec(dllexport) void ModelObjPredict_Mask(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList);
