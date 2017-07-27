// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>
#include <functional>
#include "net.h"
#include <time.h>

#define VIDEO_MODEL 0

using namespace std;
using namespace cv;

ncnn::Net squeezenet;
const float mean_vals[3] = { 104.f, 117.f, 123.f };

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);
	in.substract_mean_normalize(mean_vals, 0);

	ncnn::Extractor ex = squeezenet.create_extractor();
	ex.set_light_mode(true);

	ex.input("data", in);
	ncnn::Mat out;

	clock_t start, finish;
	start = clock();
	ex.extract("prob", out);
	finish = clock();
	double totaltime;
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("run time: %f\n", totaltime);

	cls_scores.resize(out.c);
	for (int j = 0; j<out.c; j++)
	{
		const float* prob = out.data + out.cstep * j;
		cls_scores[j] = prob[0];
	}

	return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk, vector<int>& index_result, vector<float>& score_result)
{
	// partial sort topk with index
	int size = cls_scores.size();
	std::vector< std::pair<float, int> > vec;
	vec.resize(size);
	for (int i = 0; i<size; i++)
	{
		vec[i] = std::make_pair(cls_scores[i], i);
	}

	std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(), std::greater< std::pair<float, int> >());

	// print topk and score
	for (int i = 0; i<topk; i++)
	{
		float score = vec[i].first;
		int index = vec[i].second;
		index_result.push_back(index);
		score_result.push_back(score);

		//fprintf(stderr, "%d = %f\n", index, score);
	}

	return 0;
}


static int load_labels(string path, vector<string>& labels)
{
	FILE* fp = fopen(path.c_str(), "r");

	while (!feof(fp))
	{
		char str[1024];
		fgets(str, 1024, fp);  //读取一行
		string str_s(str);

		if (str_s.length() > 0)
		{
			for (int i = 0; i < str_s.length(); i++)
			{
				if (str_s[i] == ' ')
				{
					string strr = str_s.substr(i, str_s.length() - i - 1);
					labels.push_back(strr);
					i = str_s.length();
				}
			}
		}
	}
	return 0;
}


int main(int argc, char** argv)
{
	//初始化模型，以及分类标签
	squeezenet.load_param("D:\\FaceIdentification\\ncnn-master\\examples\\squeezenet_v1.1.param");
	squeezenet.load_model("D:\\FaceIdentification\\ncnn-master\\examples\\squeezenet_v1.1.bin");
	vector<string> labels;
	load_labels("D:\\FaceIdentification\\ncnn-master\\examples\\synset_words.txt", labels);

#if VIDEO_MODEL
	VideoCapture cap(1);

	Mat frame;
	while (1)
	{
		cap >> frame;
		if (!frame.data)
			break;

		//前馈run squeezenet 网络
		std::vector<float> cls_scores;
		detect_squeezenet(frame, cls_scores);

		//查找识别结果标签
		vector<int> index;
		vector<float> score;
		print_topk(cls_scores, 3, index, score);

		//图形化显示识别结果
		for (int i = 0; i < index.size(); i++)
		{
			cv::putText(frame, labels[index[i]], Point(10, 10 + 30 * i), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 100, 0), 2, 2);
		}

		imshow("frame", frame);
		waitKey(1);
	}
#else
	//载入测试图片
	const char* imagepath = "D:\\FaceIdentification\\ncnn-master\\examples\\cat.jpg";
	cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
	if (m.empty())
	{
		fprintf(stderr, "cv::imread %s failed\n", imagepath);
		return -1;
	}

	//前馈run squeezenet 网络
	std::vector<float> cls_scores;
	detect_squeezenet(m, cls_scores);

	//查找识别结果标签
	vector<int> index;
	vector<float> score;
	print_topk(cls_scores, 3, index, score);

	//图形化显示识别结果
	for (int i = 0; i < index.size(); i++)
	{
		cv::putText(m, labels[index[i]], Point(10, 10 + 30 * i), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 100, 0), 2, 2);
	}

	imshow("m", m);
	imwrite("dog_result.jpg", m);
	waitKey(0);
#endif

	return 0;
}