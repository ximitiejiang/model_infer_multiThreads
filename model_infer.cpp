// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <string>
#include <vector>
#include "model_deploy/common/include/paddle_deploy.h"
#include "model_deploy/common/include/model_infer.h"
#include <windows.h>	// GetCurrentThreadId()

/*
* ģ�ͳ�ʼ��/ע��ӿ�
*
* model_type: ��ʼ��ģ������: det,seg,clas,paddlex
*
* model_filename: ģ���ļ�·��
*
* params_filename: �����ļ�·��
*
* cfg_file: �����ļ�·��
*
* use_gpu: �Ƿ�ʹ��GPU
*
* gpu_id: ָ����x��GPU
*
* paddlex_model_type: model_typeΪpaddlxʱ�����ص�ʵ��paddlexģ�͵�����: det, seg, clas
*/
extern "C" __declspec(dllexport) PaddleDeploy::Model * InitModel(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type)
{
	// create model
	PaddleDeploy::Model* model = PaddleDeploy::CreateModel(model_type);  //FLAGS_model_type

	// model init
	model->Init(cfg_file);

	// inference engine init
	PaddleDeploy::PaddleEngineConfig engine_config;
	engine_config.model_filename = model_filename;
	engine_config.params_filename = params_filename;
	engine_config.use_gpu = use_gpu;
	engine_config.gpu_id = gpu_id;
	bool init = model->PaddleEngineInit(engine_config);
	if (!init)
	{
		LOGC("ERR", "init model failed");
	}

	// det, seg, clas, paddlex
	if (strcmp(model_type, "paddlex") == 0) // ��paddlexģ�ͣ��򷵻ؾ���֧�ֵ�ģ������: det, seg, clas
	{
		// detector
		if (model->yaml_config_["model_type"].as<std::string>() == std::string("detector"))
		{
			strcpy(paddlex_model_type, "det");
		}
		else if (model->yaml_config_["model_type"].as<std::string>() == std::string("segmenter"))
		{
			strcpy(paddlex_model_type, "seg");
		}
		else if (model->yaml_config_["model_type"].as<std::string>() == std::string("classifier"))
		{
			strcpy(paddlex_model_type, "clas");
		}
	}
	return model;
}

// ��ʼ��ģ�ʹ�tensorRT����
extern "C" __declspec(dllexport) PaddleDeploy::Model * InitModel_TRT(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type)
{
	// create model
	PaddleDeploy::Model* model = PaddleDeploy::CreateModel(model_type);  //FLAGS_model_type

	// model init
	model->Init(cfg_file);

	// inference engine init
	PaddleDeploy::PaddleEngineConfig engine_config;
	engine_config.model_filename = model_filename;
	engine_config.params_filename = params_filename;
	engine_config.use_gpu = use_gpu;
	engine_config.gpu_id = gpu_id;

	// ʹ��tensorRT��ǿ�ƴ�gpu
	engine_config.use_gpu = true;
	engine_config.use_trt = true;

	// ���tensorRT��Ҫ����ͼ�Ż�
	engine_config.precision = 0;			// ����ѡ��Ĭ��fp32,����fp16,int8
	engine_config.min_subgraph_size = 40;	// ��С��ͼ��Խ�����Ż���Խ�ͣ�Խ��Խ���ܺ��Զ�̬ͼ
	engine_config.max_workspace_size = 1 << 30;

	// ����min,max,optim�ߴ綨��
	engine_config.min_input_shape["x"] = { 1, 3, 512, 512 };
	engine_config.max_input_shape["x"] = { 1, 3, 1024, 1024 };
	engine_config.optim_input_shape["x"] = { 1, 3, 1024, 1024 };

	// �ֱ�����С�������������ߴ磺��Ҫ����ģ������ߴ����
	// ��������ģ������Ĺؼ��ֲ�ͬ����ͨ��netron�鿴INPUTS.name������segģ��INPUTS.name=x
	// ��������ж�̬����ߴ粻ƥ��Ľڵ㣬��Ҫ�ֶ�����
	if (strcmp("clas", model_type) == 0) {
		// Adjust shape according to the actual model
		engine_config.min_input_shape["inputs"] = { 1, 3, 224, 224 };
		engine_config.max_input_shape["inputs"] = { 1, 3, 224, 224 };
		engine_config.optim_input_shape["inputs"] = { 1, 3, 224, 224 };
	}
	else if (strcmp("det", model_type) == 0) {
		// Adjust shape according to the actual model
		engine_config.min_input_shape["image"] = { 1, 3, 608, 608 };
		engine_config.max_input_shape["image"] = { 1, 3, 608, 608 };
		engine_config.optim_input_shape["image"] = { 1, 3, 608, 608 };
	}
	else if (strcmp("seg", model_type) == 0) {
		// Additional nodes need to be added, pay attention to the output prompt
		engine_config.min_input_shape["x"] = { 1, 3, 512, 512 };
		engine_config.max_input_shape["x"] = { 1, 3, 1024, 1024 };
		engine_config.optim_input_shape["x"] = { 1, 3, 1024, 1024 };

	}
	bool init = model->PaddleEngineInit(engine_config);
	if (!init)
	{
		LOGC("INFO", "init model failed");
	}
	// det, seg, clas, paddlex
	if (strcmp(model_type, "paddlex") == 0) // ��paddlexģ�ͣ��򷵻ؾ���֧�ֵ�ģ������: det, seg, clas
	{
		// detector
		if (model->yaml_config_["model_type"].as<std::string>() == std::string("detector"))
		{
			strcpy(paddlex_model_type, "det");
		}
		else if (model->yaml_config_["model_type"].as<std::string>() == std::string("segmenter"))
		{
			strcpy(paddlex_model_type, "seg");
		}
		else if (model->yaml_config_["model_type"].as<std::string>() == std::string("classifier"))
		{
			strcpy(paddlex_model_type, "clas");
		}
	}
	return model;
}


/*
* �������ӿ�
*
* img: input for predicting.
*
* nWidth: width of img.
*
* nHeight: height of img.
*
* nChannel: channel of img.
*
* output: result of pridict ,include category_id??score??coordinate??
*
* nBoxesNum?? number of box
*
* LabelList: label list of result
*/
extern "C" __declspec(dllexport) void Det_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* output, int* nBoxesNum, char* LabelList)
{
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel == 3)
	{
		nType = CV_8UC3;
	}
	else
	{
		std::cout << "Only support 3 channel image." << std::endl;
		return;
	}

	cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
	memcpy(input.data, img, nHeight * nWidth * nChannel * sizeof(uchar));
	//cv::imwrite("./1.png", input);
	imgs.push_back(std::move(input));

	// predict
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);

	// nBoxesNum[0] = results.size();  // results.size()�õ�����batch_size
	nBoxesNum[0] = results[0].det_result->boxes.size();  // �õ�����ͼƬԤ���bounding box��
	std::string label = "";
	//std::cout << "res: " << results[num] << std::endl;
	for (int i = 0; i < results[0].det_result->boxes.size(); i++)  // �õ����п������
	{
		//std::cout << "category: " << results[num].det_result->boxes[i].category << std::endl;
		label = label + results[0].det_result->boxes[i].category + " ";
		// labelindex
		output[i * 6 + 0] = results[0].det_result->boxes[i].category_id; // ����id
		// score
		output[i * 6 + 1] = results[0].det_result->boxes[i].score;  // �÷�
		//// box
		output[i * 6 + 2] = results[0].det_result->boxes[i].coordinate[0]; // x1, y1, x2, y2
		output[i * 6 + 3] = results[0].det_result->boxes[i].coordinate[1]; // ���ϡ����µĶ���
		output[i * 6 + 4] = results[0].det_result->boxes[i].coordinate[2];
		output[i * 6 + 5] = results[0].det_result->boxes[i].coordinate[3];
	}
	memcpy(LabelList, label.c_str(), strlen(label.c_str()));
}


/*
* �ָ�����ӿ�
*
* img: input for predicting.
*
* nWidth: width of img.
*
* nHeight: height of img.
*
* nChannel: channel of img.
*
* output: result of pridict ,include label_map
*/
extern "C" __declspec(dllexport) void Seg_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, unsigned char* output)
{
	//LOGC("INFO", "seg in thread id [%d]", GetCurrentThreadId());
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel == 3)
	{
		nType = CV_8UC3;
		//LOGC("INFO", "infer input img w=%d, h=%d, c=%d", nWidth, nHeight, nChannel);
	}
	else
	{
		//std::cout << "Only support 3 channel image." << std::endl;
		LOGC("ERR", "Only support 3 channel images, but got channels ", nChannel);
		return;
	}

	cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
	memcpy(input.data, img, nHeight * nWidth * nChannel * sizeof(uchar));
	//cv::imwrite("D://modelinfercpp_275.bmp", input);
	imgs.push_back(std::move(input));

	// predict
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);
	// batch�޸ģ�����Ӧ�û�õ����ص�ÿ��ͼ��label_map����ô�����Ӧ�÷ֱ���results��ÿ��ͼ��Ӧ��label_map
	std::vector<uint8_t> result_map = results[0].seg_result->label_map.data; // vector<uint8_t> -- ���map
	//LOGC("INFO", "finish infer, with result_map length=%d", result_map.size());
	// ����������������Ϸ��� -- ��vector<uint8_t>ת��unsigned char *
	memcpy(output, &result_map[0], result_map.size() * sizeof(uchar));
}


/*
* �ָ�����ӿ�batch predict
*
* img: input for predicting.
*
* nWidth: width of img.
*
* nHeight: height of img.
*
* nChannel: channel of img.
*
* output: result of pridict ,include label_map
*/
extern "C" __declspec(dllexport) void Seg_ModelBatchPredict(PaddleDeploy::Model * model, const std::vector<unsigned char*> imgs, int nWidth, int nHeight, int nChannel, std::vector<unsigned char*> output)
{
	std::vector<PaddleDeploy::Result> results;
	if (imgs.size() != output.size()) {
		LOGC("ERR", "image batch size(%d) not match with results size(%d)", imgs.size(), output.size());
	}
	// Read image
	int im_vec_size = imgs.size();
	std::vector<cv::Mat> im_vec;

	int nType = 0;
	if (nChannel == 3)
	{
		nType = CV_8UC3;
	}
	else
	{
		LOGC("ERR", "Only support 3 channel images, but got channels ", nChannel);
		return;
	}
	for (int i = 0; i < im_vec_size; i++) {
		cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
		memcpy(input.data, imgs[i], nHeight * nWidth * nChannel * sizeof(uchar));
		im_vec.emplace_back(std::move(input));
	}
	if (!model->Predict(im_vec, &results, 1)) {
		LOGC("ERR", "predict batch images failed");
	}
	// batch�޸ģ�����Ӧ�û�õ����ص�ÿ��ͼ��label_map����ô�����Ӧ�÷ֱ���results��ÿ��ͼ��Ӧ��label_map
	for (int i = 0; i < im_vec_size; i++) {
		std::vector<uint8_t> result_map = results[i].seg_result->label_map.data; // vector<uint8_t> -- ���map
		// ����������������Ϸ��� -- ��vector<uint8_t>ת��unsigned char *
		memcpy(output[i], &result_map[0], result_map.size() * sizeof(uchar));
	}
}




/*
* ʶ������ӿ�
*
* img: input for predicting.
*
* nWidth: width of img.
*
* nHeight: height of img.
*
* nChannel: channel of img.
*
* score: result of pridict ,include score
*
* category: result of pridict ,include category_string
*
* category_id: result of pridict ,include category_id
*/
extern "C" __declspec(dllexport) void Cls_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* score, char* category, int* category_id)
{
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel == 3)
	{
		nType = CV_8UC3;
	}
	else
	{
		std::cout << "Only support 3 channel image." << std::endl;
		return;
	}

	cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
	memcpy(input.data, img, nHeight * nWidth * nChannel * sizeof(uchar));
	//cv::imwrite("./1.png", input);
	imgs.push_back(std::move(input));

	// predict
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);

	*category_id = results[0].clas_result->category_id;
	// ������������������Ϸ��� -- string --> char* 
	memcpy(category, results[0].clas_result->category.c_str(), strlen(results[0].clas_result->category.c_str()));
	// �����������ֵ����
	*score = results[0].clas_result->score;
}


/*
* MaskRCNN����ӿ�
*
* img: input for predicting.
*
* nWidth: width of img.
*
* nHeight: height of img.
*
* nChannel: channel of img.
*
* box_output: result of pridict ,include label+score+bbox
*
* mask_output: result of pridict ,include label_map
*
* nBoxesNum: result of pridict ,include BoxesNum
*
* LabelList: result of pridict ,include LabelList
*/
extern "C" __declspec(dllexport) void Mask_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList)
{
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel == 3)
	{
		nType = CV_8UC3;
	}
	else
	{
		std::cout << "Only support 3 channel image." << std::endl;
		return;
	}

	cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
	memcpy(input.data, img, nHeight * nWidth * nChannel * sizeof(uchar));
	imgs.push_back(std::move(input));

	// predict  -- ��ε����������ʱ�����
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);  // ��Infer����������

	nBoxesNum[0] = results[0].det_result->boxes.size();  // �õ�����ͼƬԤ���bounding box��
	std::string label = "";

	for (int i = 0; i < results[0].det_result->boxes.size(); i++)  // �õ����п������
	{
		// �߽��Ԥ����
		label = label + results[0].det_result->boxes[i].category + " ";
		// labelindex
		box_output[i * 6 + 0] = results[0].det_result->boxes[i].category_id; // ����id
		// score
		box_output[i * 6 + 1] = results[0].det_result->boxes[i].score;  // �÷�
		//// box
		box_output[i * 6 + 2] = results[0].det_result->boxes[i].coordinate[0]; // x1, y1, x2, y2
		box_output[i * 6 + 3] = results[0].det_result->boxes[i].coordinate[1]; // ���ϡ����µĶ���
		box_output[i * 6 + 4] = results[0].det_result->boxes[i].coordinate[2];
		box_output[i * 6 + 5] = results[0].det_result->boxes[i].coordinate[3];

		//MaskԤ����
		for (int j = 0; j < results[0].det_result->boxes[i].mask.data.size(); j++)
		{
			if (mask_output[j] == 0)
			{
				mask_output[j] = results[0].det_result->boxes[i].mask.data[j];
			}
		}
	}
	memcpy(LabelList, label.c_str(), strlen(label.c_str()));
}


/*
* ģ������/ע���ӿ�
*/
extern "C" __declspec(dllexport) void DestructModel(PaddleDeploy::Model * model)
{
	if (model != NULL) {
		delete model;
		model = NULL;
	}
	if (model == NULL) LOGC("INFO", "destruct model success");
	else LOGC("ERR", "delete model failed");
}


// �������η�װ����ʼ��
void ModelWrapper::InitModelEnter(const char* model_type, const char* model_dir, int gpu_id, bool use_trt)
{
	// ��ʼ���̳߳أ�����ָ�������̣߳�ÿ���߳�ָ�����̳߳ص�һ���̺߳�
	pool = new ThreadPool(num_threads);
	pool->init();

	std::string model_filename = std::string(model_dir) + "\\model.pdmodel";
	std::string params_filename = std::string(model_dir) + "\\model.pdiparams";
	std::string cfg_file = std::string(model_dir) + "\\deploy.yaml";

	bool use_gpu = true;
	char* paddle_model_type = NULL;
	if (!use_trt) {
		_model = InitModel(model_type,
			model_filename.c_str(),
			params_filename.c_str(),
			cfg_file.c_str(),
			use_gpu,
			gpu_id,
			paddle_model_type);
	}
	else
	{
		_model = InitModel_TRT(model_type,
			model_filename.c_str(),
			params_filename.c_str(),
			cfg_file.c_str(),
			use_gpu,
			gpu_id,
			paddle_model_type);
	}
}

// �������η�װ����ͼ����
void ModelWrapper::SegPredictEnter(unsigned char* imageData, int width, int height, int channels, unsigned char* result_map)
{
	cv::Mat src;
	if (channels == 1) {
		src = cv::Mat(height, width, CV_8UC1, imageData);
		cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
	}
	else
	{
		src = cv::Mat(height, width, CV_8UC3, imageData);
	}
	int predChannels = src.channels();
	UCHAR* _imageData = src.data;
	auto future1 = pool->submit(Seg_ModelPredict, _model, _imageData, width, height, predChannels, result_map);
	future1.get();
}

// �������η�װ��ģ����Դ�ͷ�
void ModelWrapper::DestructModelEnter()
{
	// �ͷ��̳߳��������߳�
	pool->shutdown();
	if (pool != NULL) {
		delete pool;
		pool = NULL;
	}
	// �ͷ�ģ����Դ
	if (_model != NULL) {
		DestructModel(_model);
	}
}


// �������η�װ�ӿ�api
extern "C" __declspec(dllexport) ModelWrapper * ModelObjInit(const char* model_type, const char* model_dir, int gpu_id, bool use_trt)
{
	ModelWrapper* segObj = new ModelWrapper();
	segObj->InitModelEnter(model_type, model_dir, gpu_id, use_trt);
	return segObj;
}

extern "C" __declspec(dllexport) void ModelObjPredict_Seg(ModelWrapper * segObj, unsigned char* imageData, int width, int height, int channels, unsigned char* resultMap)
{
	segObj->SegPredictEnter(imageData, width, height, channels, resultMap);
}

extern "C" __declspec(dllexport) void ModelObjDestruct(ModelWrapper * segObj)
{
	// ���ͷ�ģ���ڲ�����Դ
	segObj->DestructModelEnter();
	// ���ͷŶ���ģ����Դ
	delete  segObj;
}