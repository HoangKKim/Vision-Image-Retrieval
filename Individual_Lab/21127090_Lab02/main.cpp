#include "function.h"

int main(int n_args, char* args[]) {

	if (n_args != 3) {
		cout << "Invalid action";
		return 0;
	}
	String inputPathFile = args[1];
	String outputPathFile = args[2];

	Image MyImage(inputPathFile);
	Mat image = MyImage.getImage();

	// check the empty image
	if (image.empty()) {
		cout << "Cannot open image";
		return 0; 
	}

	// feature extraction
	Mat preResult, result;
	vector<KeyPoint> keypoints_SIFT, keypoints_ORB;
	keypoints_SIFT = MyImage.featureExtractionBySIFT(image);
	keypoints_ORB = MyImage.featureExtractionByORB(image);

	// draw keypoints
	drawKeypoints(image, keypoints_SIFT, preResult, Scalar(0, 0, 255));
	drawKeypoints(preResult, keypoints_ORB, result, Scalar(255, 0, 0));

	// show the result and save it
	imshow("Feature Extraction", result);
	imwrite(outputPathFile, result);
	waitKey();
}



