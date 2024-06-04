#include "function.h"

// constructor and destructor
Video::Video(String video_filePath, bool usingWebcam) {
	this->video_filePath = video_filePath;
	this->usingWebcam = usingWebcam;
};

Video::~Video() {
	if (video.isOpened()) {
		video.release();
	}
};

Mat Video::getFrame() {
	this->video >> frame;
	return frame;
};

void Video::openWebcam() {
	this->video.open(0);
}

void Video::openVideo() {
	this->video.open(this->video_filePath);
	if (!this->video.isOpened()) {
		cout << "Can not open the video!" << endl;
		openWebcam();
	}
}

Mat Video::drawHistogram_RGB(Mat image, int hist_w, int hist_h) {
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

	// create an image to display the histogram
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImg(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

	// normalize the color histogram that the tallest bar fits within the height of the display image.
	normalize(hist_b, hist_b, 0, histImg.rows, NORM_MINMAX, -1, Mat());
	normalize(hist_g, hist_g, 0, histImg.rows, NORM_MINMAX, -1, Mat());
	normalize(hist_r, hist_r, 0, histImg.rows, NORM_MINMAX, -1, Mat());


	// draw histogram
	for (int i = 1; i < histSize; i++)
	{
		line(histImg, Point(bin_w * (i - 1), hist_h - cvRound(hist_b.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(hist_b.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImg, Point(bin_w * (i - 1), hist_h - cvRound(hist_g.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(hist_g.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImg, Point(bin_w * (i - 1), hist_h - cvRound(hist_r.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(hist_r.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	return histImg;
}

void Video::showResult(Mat frame, Mat histImg) {
	imshow("Origin image", frame);

	imshow("Histogram", histImg);
}