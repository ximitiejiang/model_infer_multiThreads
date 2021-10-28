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
* 模型初始化/注册接口
*
* model_type: 初始化模型类型: det,seg,clas,paddlex
*
* model_filename: 模型文件路径
*
* params_filename: 参数文件路径
*
* cfg_file: 配置文件路径
*
* use_gpu: 是否使用GPU
*
* gpu_id: 指定第x号GPU
*
* paddlex_model_type: model_type为paddlx时，返回的实际paddlex模型的类型: det, seg, clas
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
	else
	{
		LOGC("INFO", "init model successfully: use_gpu=%d, gpu_id=%d, model path=%s", use_gpu, gpu_id, model_filename);
	}

	// det, seg, clas, paddlex
	if (strcmp(model_type, "paddlex") == 0) // 是paddlex模型，则返回具体支持的模型类型: det, seg, clas
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

// 初始化模型带tensorRT加速
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

	// 使用tensorRT则强制打开gpu
	engine_config.use_gpu = true;
	engine_config.use_trt = true;

	// 针对tensorRT需要做子图优化
	engine_config.precision = 0;			// 精度选择，默认fp32,还有fp16,int8
	engine_config.min_subgraph_size = 40;	// 最小子图，越大则优化度越低，越大越可能忽略动态图
	engine_config.max_workspace_size = 1 << 30;

	// 增加min,max,optim尺寸定义
	engine_config.min_input_shape["x"] = { 1, 3, 512, 512 };
	engine_config.max_input_shape["x"] = { 1, 3, 1024, 1024 };
	engine_config.optim_input_shape["x"] = { 1, 3, 1024, 1024 };

	// 分别定义最小、最大、最优输入尺寸：需要根据模型输入尺寸调整
	// 这里三种模型输入的关键字不同，可通过netron查看INPUTS.name，比如seg模型INPUTS.name=x
	// 另外如果有动态输入尺寸不匹配的节点，需要手动定义
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
	if (strcmp(model_type, "paddlex") == 0) // 是paddlex模型，则返回具体支持的模型类型: det, seg, clas
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
* 检测推理接口
*
* img: input for predicting.
*
* nWidth: width of img.
*
* nHeight: height of img.
*
* nChannel: channel of img.
*
* output: result of pridict ,include category_id£¬score£¬coordinate¡£
*
* nBoxesNum£º number of box
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

	// nBoxesNum[0] = results.size();  // results.size()得到的是batch_size
	nBoxesNum[0] = results[0].det_result->boxes.size();  // 得到单张图片预测的bounding box数
	std::string label = "";
	//std::cout << "res: " << results[num] << std::endl;
	for (int i = 0; i < results[0].det_result->boxes.size(); i++)  // 得到所有框的数据
	{
		//std::cout << "category: " << results[num].det_result->boxes[i].category << std::endl;
		label = label + results[0].det_result->boxes[i].category + " ";
		// labelindex
		output[i * 6 + 0] = results[0].det_result->boxes[i].category_id; // 类别的id
		// score
		output[i * 6 + 1] = results[0].det_result->boxes[i].score;  // 得分
		//// box
		output[i * 6 + 2] = results[0].det_result->boxes[i].coordinate[0]; // x1, y1, x2, y2
		output[i * 6 + 3] = results[0].det_result->boxes[i].coordinate[1]; // 左上、右下的顶点
		output[i * 6 + 4] = results[0].det_result->boxes[i].coordinate[2];
		output[i * 6 + 5] = results[0].det_result->boxes[i].coordinate[3];
	}
	memcpy(LabelList, label.c_str(), strlen(label.c_str()));
}


/*
* 分割推理接口
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
	// batch修改：这里应该会得到返回的每张图的label_map，那么下面就应该分别处理results中每张图对应的label_map
	std::vector<uint8_t> result_map = results[0].seg_result->label_map.data; // vector<uint8_t> -- 结果map
	//LOGC("INFO", "finish infer, with result_map length=%d", result_map.size());
	// 拷贝输出结果到输出上返回 -- 将vector<uint8_t>转成unsigned char *
	memcpy(output, &result_map[0], result_map.size() * sizeof(uchar));
}


/*
* 分割推理接口batch predict
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
	// batch修改：这里应该会得到返回的每张图的label_map，那么下面就应该分别处理results中每张图对应的label_map
	for (int i = 0; i < im_vec_size; i++) {
		std::vector<uint8_t> result_map = results[i].seg_result->label_map.data; // vector<uint8_t> -- 结果map
		// 拷贝输出结果到输出上返回 -- 将vector<uint8_t>转成unsigned char *
		memcpy(output[i], &result_map[0], result_map.size() * sizeof(uchar));
	}
}




/*
* 识别推理接口
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
	cv::imwrite("D:\\1.png", input);
	imgs.push_back(std::move(input));

	// predict
	std::vector<PaddleDeploy::Result> results;
	//LOGC("INFO", "begin predict");
	model->Predict(imgs, &results, 1);
	//LOGC("INFO", "got pred result: score=%f", results[0].clas_result->score);
	//LOGC("INFO", "got pred result: category_id=%d", results[0].clas_result->category_id);
	//LOGC("INFO", "got pred result: category=%s", results[0].clas_result->category);
	*category_id = results[0].clas_result->category_id;
	// 拷贝输出类别结果到输出上返回 -- string --> char* 
	memcpy(category, results[0].clas_result->category.c_str(), strlen(results[0].clas_result->category.c_str()));
	// 拷贝输出概率值返回
	*score = results[0].clas_result->score;
}


/*
* MaskRCNN推理接口
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

	// predict  -- 多次点击单张推理时会出错
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);  // 在Infer处发生错误

	nBoxesNum[0] = results[0].det_result->boxes.size();  // 得到单张图片预测的bounding box数
	std::string label = "";

	for (int i = 0; i < results[0].det_result->boxes.size(); i++)  // 得到所有框的数据
	{
		// 边界框预测结果
		label = label + results[0].det_result->boxes[i].category + " ";
		// labelindex
		box_output[i * 6 + 0] = results[0].det_result->boxes[i].category_id; // 类别的id
		// score
		box_output[i * 6 + 1] = results[0].det_result->boxes[i].score;  // 得分
		//// box
		box_output[i * 6 + 2] = results[0].det_result->boxes[i].coordinate[0]; // x1, y1, x2, y2
		box_output[i * 6 + 3] = results[0].det_result->boxes[i].coordinate[1]; // 左上、右下的顶点
		box_output[i * 6 + 4] = results[0].det_result->boxes[i].coordinate[2];
		box_output[i * 6 + 5] = results[0].det_result->boxes[i].coordinate[3];

		//Mask预测结果
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
* 模型销毁/注销接口
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


// 新增二次封装：初始化
void ModelWrapper::InitModelEnter(const char* model_type, const char* model_dir, int gpu_id, bool use_trt)
{
	// 初始化线程池：创建指定个数线程，每个线程指定到线程池的一个线程号
	pool = new ThreadPool(num_threads);
	pool->init();

	std::string model_filename = std::string(model_dir) + "\\model.pdmodel";
	std::string params_filename = std::string(model_dir) + "\\model.pdiparams";
	std::string cfg_file = std::string(model_dir) + "\\deploy.yaml";

	bool use_gpu = true;
	char* paddle_model_type = NULL;
	if (!use_trt) {
		_model = InitModel(model_type,
			model_filename.c_str(),    // *.pdmodel
			params_filename.c_str(),   // *.pdiparams
			cfg_file.c_str(),          // *.yaml 
			use_gpu,
			gpu_id,
			paddle_model_type);
	}
	else
	{
		_model = InitModel_TRT(model_type,
			model_filename.c_str(),    // *.pdmodel
			params_filename.c_str(),   // *.pdiparams
			cfg_file.c_str(),          // *.yaml 
			use_gpu,
			gpu_id,
			paddle_model_type);
	}
}

// 新增二次封装：单图推理
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

// 检测模型
void ModelWrapper::DetPredictEnter(unsigned char* imageData, int width, int height, int channels, float* output, int* nBoxesNum, char* LabelList)
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
	auto future1 = pool->submit(Det_ModelPredict, _model, _imageData, width, height, predChannels, output, nBoxesNum, LabelList);
	future1.get();
}

// 分类模型
void ModelWrapper::ClsPredictEnter(unsigned char* imageData, int width, int height, int channels, float* score, char* category, int* category_id)
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
	auto future1 = pool->submit(Cls_ModelPredict, _model, _imageData, width, height, predChannels, score, category, category_id);
	future1.get();
}

// Mask模型
void ModelWrapper::MaskPredictEnter(unsigned char* imageData, int width, int height, int channels, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList)
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
	auto future1 = pool->submit(Mask_ModelPredict, _model, _imageData, width, height, predChannels, box_output, mask_output, nBoxesNum, LabelList);
	future1.get();
}


// 新增二次封装：模型资源释放
void ModelWrapper::DestructModelEnter()
{
	// 释放线程池中所有线程
	pool->shutdown();
	if (pool != NULL) {
		delete pool;
		pool = NULL;
	}
	// 释放模型资源
	if (_model != NULL) {
		DestructModel(_model);
	}
}


// 新增二次封装接口api
extern "C" __declspec(dllexport) ModelWrapper * ModelObjInit(const char* model_type, const char* model_dir, int gpu_id, bool use_trt)
{
	ModelWrapper* modelObj = new ModelWrapper();
	modelObj->InitModelEnter(model_type, model_dir, gpu_id, use_trt);
	return modelObj;
}


extern "C" __declspec(dllexport) void ModelObjDestruct(ModelWrapper * modelObj)
{
	// 先释放模型内部的资源
	modelObj->DestructModelEnter();
	// 再释放堆区模型资源
	delete  modelObj;
}

extern "C" __declspec(dllexport) void ModelObjPredict_Seg(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, unsigned char* resultMap)
{
	modelObj->SegPredictEnter(imageData, width, height, channels, resultMap);
}

extern "C" __declspec(dllexport) void ModelObjPredict_Det(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* output, int* nBoxesNum, char* LabelList)
{
	modelObj->DetPredictEnter(imageData, width, height, channels, output, nBoxesNum, LabelList);
}

extern "C" __declspec(dllexport) void ModelObjPredict_Cls(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* score, char* category, int* category_id)
{
	modelObj->ClsPredictEnter(imageData, width, height, channels, score, category, category_id);
}

extern "C" __declspec(dllexport) void ModelObjPredict_Mask(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList)
{
	modelObj->MaskPredictEnter(imageData, width, height, channels, box_output, mask_output, nBoxesNum, LabelList);
}