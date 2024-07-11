#include "train_function.h";
#include "query_function.h"

// 21127090.exe <test_txt (path_file)> <database (CD / TMBuD)> <feature_type (SIFT, ORB, Color, Corre)> <kImages> <showImagesResult>

int main(int nargs, char* args[]) {
	// this line is use to call the preprocess data function to create database
	//preprocess_data();

	if (nargs != 6) {
		cout << "Invalid input" << endl;
		return 0;
	}

	string query_filename = args[1];
	string data_type = args[2];
	string feature_type = args[3];
	int kImages = stoi(args[4]);
	string flag = args[5];

	string data_filename;
	string centroids_filename;
	string labels_filename;
	if (data_type == "CD") {
		data_filename = "../../Database/CD/CD.xml";
		centroids_filename = "../../Database/CD/Centroid_" + feature_type + "_100.bin";
		labels_filename = "../../Database/CD/labels.txt";
	}
	else if (data_type == "TMBuD") {
		data_filename = "../../Database/TMBuD/TMBuD.xml";
		centroids_filename = "../../Database/TMBuD/Centroid_" + feature_type + "_100.bin";
		labels_filename = "../../Database/TMBuD/labels.txt";
	}

	time_t start, end;

	// load corresponding database (data & centroids)
	Searcher mySearcher;

	mySearcher.setDatabase(data_filename, data_type, feature_type, centroids_filename, labels_filename);

	// load query images
	Query myQuery;
	myQuery.loadQueryImages(query_filename);
	system("cls");

	for (int i = 0; i < myQuery.getImages().size(); i++) {
		cout << "Retriving ... " << endl;
		time(&start);

		Mat queryImage = imread(myQuery.getImages()[i].first);
		string queryLabel = myQuery.getImages()[i].second;

		// search by feature 
		vector<pair<int, double>> distances;
		if (feature_type == "SIFT") {
			distances = mySearcher.searchBySIFT(queryImage);
		}
		else if (feature_type == "ORB") {
			distances = mySearcher.searchByORB(queryImage);
		}
		else if (feature_type == "Histogram") {
			distances = mySearcher.searchByHistogram(queryImage);
		}
		// get k closet images
		mySearcher.setResultImages(kImages, distances);
		time(&end);

		if (flag == "true") {
			mySearcher.showResult(queryImage, kImages);
		}

		double duration = difftime(end, start);
		double AP = mySearcher.computeAveragePrecision(queryLabel);
		double recall = mySearcher.computeRecall(queryLabel);

		mySearcher.printResultInfor(AP, recall, kImages, duration);
		mySearcher.addAveragePrecision(AP);
		mySearcher.addRecall(recall);

		mySearcher.removeResultImages();
	}

	cout << endl << "-------------------------------------------------------------" << endl;
	cout << "TOTAL RESULT FOR RETRIEVING " << endl << endl;;
	double sum = 0;
	for (int i = 0; i < mySearcher.getAveragePrecision().size(); i++) {
		sum += mySearcher.getAveragePrecision()[i];
		cout << "Image: " << myQuery.getImages()[i].first << " - " << "AP: " << mySearcher.getAveragePrecision()[i] << " - " << "Recall: " << mySearcher.getRecall()[i] << endl;
	}
	cout << endl << "- mAP: " << (double)sum / (double)mySearcher.getAveragePrecision().size();



}