#include "function.h"

Feature::Feature() {
};

void Feature::setFeature(vector<Mat> descriptors, Mat histograms) {
	this->descriptors = descriptors;
	this->histograms = histograms;
}

void Feature::addDescriptor(Mat descriptor) {
	this->descriptors.push_back(descriptor);
}

void Feature::addHistogram(Mat histogram) {
	this->histograms.push_back(histogram);
}

Mat Feature::getHistograms() {
	return this->histograms;
}

vector<Mat> Feature::getDescriptor() {
	return this->descriptors;
}

Mat Feature::extractFeature_SIFT(Mat image) {
	Mat grayImg;

	// convert to grayImg
	cvtColor(image, grayImg, COLOR_BGR2GRAY);

	// extract features using SIFT
	vector<KeyPoint> keypoints;
	Mat descriptor;

	Ptr<Feature2D> sift = SIFT::create();
	// Detect keypoints and compute descriptors
	sift->detectAndCompute(grayImg, Mat(), keypoints, descriptor);
	return descriptor;
};

Mat Feature::computeHistogram(Mat image) {
	int histSize = 256;		// set the number of bins in the histogram
	float range[] = { 0, 256 };		// define the range of the histogram values
	const float* histRange[] = { range };
	bool uniform = true, accumulate = false;	// Specifies the properties for the histogram calculation: uniform bins and no accumulation of results
	Mat hist_b, hist_g, hist_r;
	vector<Mat> bgr_planes;

	// using split to separate the image in its three R, G and B planes
	split(image, bgr_planes);

	// compute the histogram for each color channel
	calcHist(&bgr_planes[0], 1, 0, Mat(), hist_b, 1, &histSize, histRange, uniform, accumulate);		// for blue 
	calcHist(&bgr_planes[1], 1, 0, Mat(), hist_g, 1, &histSize, histRange, uniform, accumulate);		// for green
	calcHist(&bgr_planes[2], 1, 0, Mat(), hist_r, 1, &histSize, histRange, uniform, accumulate);		// for red

	normalize(hist_b, hist_b, 0, 1, NORM_MINMAX, -1, Mat());
	normalize(hist_g, hist_g, 0, 1, NORM_MINMAX, -1, Mat());
	normalize(hist_r, hist_r, 0, 1, NORM_MINMAX, -1, Mat());

	Mat histImg;
	hconcat(hist_b.t(), hist_g.t(), histImg);
	hconcat(histImg, hist_r.t(), histImg);

	return histImg;
}

