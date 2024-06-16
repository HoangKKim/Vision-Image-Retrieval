#include "function.h"

Mat BagOfWord_Model::getVisualWords() {
	return this->visualWords;
}

void BagOfWord_Model::buildVisualWords(int k, vector<Mat> listDescriptors) {
	Mat allDescriptor;
	// add all descriptors into a Mat
	for (const auto& descriptor : listDescriptors) {
		allDescriptor.push_back(descriptor);
	}
	// apply K-means on all descriptors
	Mat labels, centers;
	kmeans(allDescriptor, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
	// center is visual words in BOW model
	this->visualWords = centers;
}
vector<Mat> BagOfWord_Model::computeHistogramOfVisualWords(const vector<Mat>& descriptors, const Mat& vocabulary) {
	vector<Mat> histograms;
	for (const auto& descriptor : descriptors) {
		// create a Mat that contains number of apperance of visual words
		Mat histogram = Mat::zeros(1, vocabulary.rows, CV_32F);
		
		for (int i = 0; i < descriptor.rows; ++i) {
			// retrive each descriptor in an image
			Mat feature = descriptor.row(i);
			double minDist = DBL_MAX;
			int bestMatch = 0;

			for (int j = 0; j < vocabulary.rows; ++j) {
				// compute distance between feature and each visual word by NORM_L2
				double dist = norm(feature - vocabulary.row(j), NORM_L2);
				if (dist < minDist) {
					minDist = dist;
					bestMatch = j;
				}
			}
			// increase number of apperance of the corresponding VW
			histogram.at<float>(bestMatch)++;
		}
		histograms.push_back(histogram);
	}
	return histograms;
}