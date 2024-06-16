#include "function.h"

int main(int n_args, char* args[]) {
	if (n_args != 5) {
		cout << "Invalid action";
		return 0;
	}

	string reqs = args[1];
	int kCluster = stoi(args[4]);
	String train_Path = args[2];
	String test_Path = args[3];
	// get dataset
	//Dataset myDataset("../training_images/*.jpg", "../TestImages/*.jpg");
	Dataset myDataset(train_Path, test_Path);

	Feature train_features;

	// compute histogram & SIFT
	myDataset.computeAllTrainFeatures();
	train_features = myDataset.getTrainFeatures();

	if (reqs == "1") {
		// ------------------------------- yc1 ------------------------
		Mat labels, centers;

		// clustering by histogram
		kmeans(train_features.getHistograms(), kCluster, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), INTER_MAX, KMEANS_PP_CENTERS, centers);
		vector <vector <Mat>> imageInCluster(kCluster);

		// save image in a same cluster in a vector<Mat>
		for (int i = 0; i < labels.rows; i++) {
			imageInCluster[labels.at<int>(i)].push_back(myDataset.getTrainImages()[i]);
		}


		// display all images in a cluster in a window 
		for (int i = 0; i < imageInCluster.size(); i++) {
			Mat canvas = makeCanvas(imageInCluster[i], 1000, 6);
			imshow("Cluster " + to_string(i + 1), canvas);

			waitKey();
			imwrite("k=" + to_string(kCluster) + "_Cluster" + to_string(i + 1) + ".jpg", canvas);
		}
	}
	else if(reqs == "2") {
		// ------------------------------ yc2 ----------------------------
		BagOfWord_Model myBOW_model;

		Feature test_features;
		myDataset.computeTestFeatures_SIFT();
		test_features = myDataset.getTestFeatures();

		// build visual words
		myBOW_model.buildVisualWords(kCluster, train_features.getDescriptor());

		cout << "Compute Histogram of Visual Words with number of word is " << kCluster << endl << endl;
		vector<Mat> histograms = myBOW_model.computeHistogramOfVisualWords(test_features.getDescriptor(), myBOW_model.getVisualWords());

		int i = 1;
		for (const auto& hist : histograms) {
			cout << "Image" << i << endl;
			cout << hist << endl;
			cout << endl;
			i++;
		}
		cout << endl << endl;
		
		waitKey();
	}	
}

