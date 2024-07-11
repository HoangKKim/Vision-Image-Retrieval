#include "train_function.h"

string Image::getFileName() {
    return filename;
}

string Image::getLabel() {
    return label;
}

Mat Image::getSIFT() {
    return sift;
}

Mat Image::getORB() {
    return orb;
}

Mat Image::getHistogram() {
    return histogram;
}

Mat Image::getCorrelogram() {
    return correlogram;
}

// Implementation of setters
void Image::setFileName(string filename) {
    this->filename = filename;
}

void Image::setLabel(string label) {
    this->label = label;
}

void Image::setSIFT(Mat sift) {
    this->sift = sift;
}

void Image::setORB(Mat orb) {
    this->orb = orb;
}

void Image::setHistogram(Mat histogram) {
    this->histogram = histogram;
}

void Image::setCorrelogram(Mat correlogram) {
    this->correlogram = correlogram;
}

Mat Image::extractLocalFeature(Mat image, string type) {
    // convert to gray image
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // init Feature Extraction Method
    Mat descriptor;
    vector<KeyPoint> keypoint;
    Ptr<Feature2D> feature;
    if (type == "SIFT") {
        feature = SIFT::create();
    }
    else if (type == "ORB") {
        feature = ORB::create();
    }
    feature->detectAndCompute(grayImage, Mat(), keypoint, descriptor);

    // compute
    if (descriptor.type() != CV_32F) {
        descriptor.convertTo(descriptor, CV_32F);
    }
    return descriptor;
}

Mat Image::computeColorHistogram(Mat image) {
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
    hconcat(hist_b, hist_g, histImg);
    hconcat(histImg, hist_r, histImg);

    return histImg;
}

Mat Image::computeCorrelogram(const Mat& image) {
    int numColors = 256;  // Assume quantization to 256 levels
    int maxDistance = 1;

    // Convert to HSV color space
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    // Quantize the HSV values to a fixed number of colors
    Mat quantized(hsv.size(), CV_32SC3);
    hsv.convertTo(quantized, CV_32SC3, numColors / 256.0);

    // Initialize the correlogram
    Mat correlogram = Mat::zeros(1, numColors * numColors * numColors * maxDistance * maxDistance, CV_32F);

    // Compute the correlogram (simplified version)
    for (int i = 0; i < hsv.rows; ++i) {
        for (int j = 0; j < hsv.cols; ++j) {
            Vec3i color = quantized.at<Vec3i>(i, j);
            int colorIndex = color[0] * numColors * numColors + color[1] * numColors + color[2];
            for (int di = -maxDistance; di <= maxDistance; ++di) {
                for (int dj = -maxDistance; dj <= maxDistance; ++dj) {
                    if (di == 0 && dj == 0) continue;
                    int ni = i + di;
                    int nj = j + dj;
                    if (ni >= 0 && ni < hsv.rows && nj >= 0 && nj < hsv.cols) {
                        Vec3i neighborColor = quantized.at<Vec3i>(ni, nj);
                        int neighborColorIndex = neighborColor[0] * numColors * numColors + neighborColor[1] * numColors + neighborColor[2];
                        int index = colorIndex * maxDistance * maxDistance + abs(di) * maxDistance + abs(dj);
                        correlogram.at<float>(index)++;
                    }
                }
            }
        }
    }

    // Normalize the correlogram
    normalize(correlogram, correlogram, 1, 0, NORM_L1, -1, Mat());

    return correlogram;
}
