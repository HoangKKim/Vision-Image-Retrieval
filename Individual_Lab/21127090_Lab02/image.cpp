#include "function.h"

// constructor - destructor
Image::Image(String inputPathFile) {
	this->image = imread(inputPathFile);
}

// get image attribute
Mat Image::getImage() {
	return this->image;
}

vector<KeyPoint> Image::featureExtractionBySIFT(Mat image) {
	Mat grayImg;

	// convert to grayImg
	cvtColor(image, grayImg, COLOR_BGR2GRAY);

	// extract features using SIFT
	vector<KeyPoint> keypoints;
	Mat descriptor;

	Ptr<Feature2D> sift = SIFT::create();
	// Detect keypoints and compute descriptors
	sift->detectAndCompute(grayImg, Mat(), keypoints, descriptor);
	return keypoints;
}

vector<KeyPoint> Image::featureExtractionByORB(Mat image) {
	Mat grayImg;

	// convert to gray image
	cvtColor(image, grayImg, COLOR_BGR2GRAY);

	// extract features using ORB
	vector<KeyPoint> keypoints;
	Mat descriptor;

	Ptr<Feature2D> orb = ORB::create();
	// Detect keypoints and compute descriptors
	orb->detectAndCompute(grayImg, Mat(), keypoints, descriptor);
	return keypoints;
}