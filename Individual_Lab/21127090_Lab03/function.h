#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class Feature {
private:
	vector<Mat> descriptors;
	Mat histograms;
public:
	Feature();
	void setFeature(vector<Mat> descriptors, Mat histograms);
	Mat computeHistogram(Mat image);
	Mat extractFeature_SIFT(Mat image);
	void addDescriptor(Mat descriptor);
	void addHistogram(Mat histogram);	
	Mat getHistograms();
	vector<Mat> getDescriptor();
};

class Dataset {

private:
	vector<Mat> train_images;
	vector<Mat> test_images;
	Feature train_features;
	Feature test_features;

public:
	Dataset(string trainFolder_Path, string testFolder_Path);
	vector<Mat> getTrainImages();
	vector<Mat> getTestImages();
	void computeAllTrainFeatures();
	void computeTestFeatures_SIFT();
	Feature getTrainFeatures();
	Feature getTestFeatures();
};

class BagOfWord_Model {
private:
	Mat visualWords;
public:
	Mat getVisualWords();
	void buildVisualWords(int k, vector<Mat> listDescriptors);
	vector<Mat> computeHistogramOfVisualWords(const vector<Mat>& descriptors, const Mat& vocabulary);

};

Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows);


