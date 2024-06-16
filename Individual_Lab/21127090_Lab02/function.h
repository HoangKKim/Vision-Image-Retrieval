#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class Image {
private:
	Mat image;
public:
	Image(String inputPathFile);
	Mat getImage();
	vector<KeyPoint> featureExtractionBySIFT(Mat image);
	vector<KeyPoint> featureExtractionByORB(Mat image);
};
