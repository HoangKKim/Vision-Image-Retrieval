#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class Video {
private:
	Mat frame;
	String video_filePath;
	VideoCapture video;
	bool usingWebcam;
public:
	// constructor - destructor
	Video(String video_filePath, bool usingWebcam);
	~Video();

	Mat getFrame();
	void openWebcam();
	void openVideo();
	// draw histogram with rgb image
	Mat drawHistogram_RGB(Mat image, int hist_w, int hist_h);
	// show a frame and the corresponding histogram 
	void showResult(Mat frame, Mat histImg);	
};