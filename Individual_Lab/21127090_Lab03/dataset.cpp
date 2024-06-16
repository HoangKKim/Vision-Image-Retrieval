#include "function.h"

Dataset::Dataset(string trainFolder_Path, string testFolder_Path) {
	vector <String> fileName;

	// read all images in training folder
	glob(trainFolder_Path, fileName, false);
	for (size_t i = 0; i < fileName.size(); i++) {
		Mat image = imread(fileName[i]);
		if (!image.empty()) {
			this->train_images.push_back(image);
		}
		// no image in folder
		else {
			break;
		}
	}

	glob(testFolder_Path, fileName, false);
	for (size_t i = 0; i < fileName.size(); i++) {
		Mat image = imread(fileName[i]);
		if (!image.empty()) {
			this->test_images.push_back(image);
		}
		// no image in folder
		else {
			break;
		}
	}
}

vector<Mat> Dataset::getTrainImages() {
	return this->train_images;
}

vector<Mat> Dataset::getTestImages() {
	return this->test_images;
}


void Dataset::computeAllTrainFeatures() {
	// compute histogram & SIFT
	Feature myFeature;
	for (const auto& image : train_images) {
		Mat hist = myFeature.computeHistogram(image);
		Mat	descriptor = myFeature.extractFeature_SIFT(image);
		
		myFeature.addHistogram(hist);
		myFeature.addDescriptor(descriptor);
		cout << '.';
	}
	cout << endl;

	Mat convertHistogram = myFeature.getHistograms();
	convertHistogram.convertTo(convertHistogram, CV_32F);
	myFeature.setFeature(myFeature.getDescriptor(), convertHistogram);
	this->train_features = myFeature;
}

Feature Dataset::getTrainFeatures() {
	return this->train_features;
}

Feature Dataset::getTestFeatures() {
	return this->test_features;
}

void Dataset::computeTestFeatures_SIFT() {
	cout << "Feature Extraction by SIFT on test images" << endl;
	Feature myFeature;

	for (const auto& image : this->test_images) {
		Mat	descriptor = myFeature.extractFeature_SIFT(image);
		myFeature.addDescriptor(descriptor);
		cout << '.';
	}
	cout << endl;
	this->test_features = myFeature;
}