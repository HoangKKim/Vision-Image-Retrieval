#include "train_function.h"

// write txt file

void countLabels(string data_filename, string data_type, string save_file) {
	Dataset data;
	data.readXML(data_filename, data_type);
	// count label
	vector<pair<string, int>> LabelsInfo;
	int count = 1;
	string previousLabel = data.getData()[0].getLabel();
	for (int i = 1; i < data.getData().size(); i++) {
		string currentLabel = data.getData()[i].getLabel();
		if (currentLabel != previousLabel) {
			LabelsInfo.push_back(make_pair(previousLabel, count));
			count = 1;
			previousLabel = currentLabel;
		}
		else {
			count++;
		}
	}
	// save file
	ofstream file(save_file);

	// Cannot open file
	if (!file.is_open()) {
		cout << "Cannot open file" << endl;
		return;
	}

	// write label for each image
	for (const auto& pair : LabelsInfo) {
		file << pair.second << " " << pair.first << endl;
	}
	//close file
	file.close();
}

void preprocess_data() {
	string csvFileName = "DATASET\ SPLIT.csv";
	string folder_TMBuD = "../../Data/TMBuD-main/TMBuD-main";
	string folder_CD = "../../Data/CD_data/training_set/training_images";

	Dataset data_CD, data_TMBuD;

	//process the initial data
	data_TMBuD.readCSV(folder_TMBuD, csvFileName);
	data_CD.createLabel(folder_CD);

	// count number of each label in a dataset
	data_TMBuD.writeXML(data_TMBuD.getData(), "../../Database/TMBuD/TMBuD.xml", "TMBuD");
	data_CD.writeXML(data_CD.getData(), "../../Database/CD/CD.xml", "CD");
	countLabels("../../Database/CD/CD.xml", "CD", "../../Database/CD/labels.txt");
	countLabels("../../Database/TMBuD/TMBuD.xml", "TMBuD", "../../Database/TMBuD/labels.txt");


	//process data in CD file
	Mat sift_descriptors, orb_descriptors, color_hist_descriptors;
	// extract feature
	for (int i = 0; i < data_CD.getData().size(); i++) {
		Image myImage = data_CD.getData()[i];
		// read image
		Mat image = imread(myImage.getFileName());
		// resize image
		resize(image, image, Size(), 0.5, 0.5); // Resize decrease 50%

		Mat sift = myImage.extractLocalFeature(image, "SIFT");
		Mat orb = myImage.extractLocalFeature(image, "ORB");
		Mat color_hist = myImage.computeColorHistogram(image);
		myImage.setSIFT(sift);
		myImage.setORB(orb);
		myImage.setHistogram(color_hist);

		data_CD.setImageInData(i, myImage);
		sift_descriptors.push_back(sift);
		orb_descriptors.push_back(orb);
		color_hist_descriptors.push_back(color_hist);
		cout << i + 1 << "..." << " ";
	}

	// find visual words
	cout << "\nKmeans..." << endl;
	Mat center_sift = buildVisualWords(sift_descriptors, 100, Mat());
	Mat center_orb = buildVisualWords(orb_descriptors, 100, Mat());
	Mat center_hist = buildVisualWords(color_hist_descriptors, 100, Mat());

	// save 
	cout << "Save data.. " << endl;
	data_CD.save_BinaryFile(center_sift, "../../Database/CD/Centroid_SIFT_100.bin");
	data_CD.save_BinaryFile(center_orb, "../../Database/CD/Centroid_ORB_100.bin");
	data_CD.save_BinaryFile(center_hist, "../../Database/CD/Centroid_hist_100.bin");

	// compute hist of visual words 
	for (int i = 0; i < data_CD.getData().size(); i++) {
		Image myImage_ = data_CD.getData()[i];
		data_CD.setCentroids(center_sift);
		Mat histBoW_sift = computeHistogram(myImage_.getSIFT(), data_CD.getCentroids());
		data_CD.setCentroids(center_orb);
		Mat histBoW_orb = computeHistogram(myImage_.getORB(), data_CD.getCentroids());
		data_CD.setCentroids(center_hist);
		Mat histBoW_hist = computeHistogram(myImage_.getHistogram(), data_CD.getCentroids());

		myImage_.setSIFT(histBoW_sift);
		myImage_.setORB(histBoW_orb);
		myImage_.setHistogram(histBoW_hist);

		data_CD.setImageInData(i, myImage_);
		cout << "." << " ";
	}

	data_CD.writeXML(data_CD.getData(), "../../Database/CD/CD.xml", "CD");

	// ----------------------------- TMBuD ------------------------------------
	for (int i = 0; i < data_TMBuD.getData().size(); i++) {
		Image myImage = data_TMBuD.getData()[i];
		// read image
		Mat image = imread(myImage.getFileName());
		// resize image
		resize(image, image, Size(), 0.5, 0.5); // Resize decrease 50%

		Mat sift = myImage.extractLocalFeature(image, "SIFT");
		Mat orb = myImage.extractLocalFeature(image, "ORB");
		Mat color_hist = myImage.computeColorHistogram(image);
		myImage.setSIFT(sift);
		myImage.setORB(orb);
		myImage.setHistogram(color_hist);

		data_TMBuD.setImageInData(i, myImage);
		sift_descriptors.push_back(sift);
		orb_descriptors.push_back(orb);
		color_hist_descriptors.push_back(color_hist);
		cout << i + 1 << "..." << " ";
	}

	// find visual words
	cout << "\nKmeans..." << endl;
	center_sift = buildVisualWords(sift_descriptors, 100, Mat());
	center_orb = buildVisualWords(orb_descriptors, 100, Mat());
	center_hist = buildVisualWords(color_hist_descriptors, 100, Mat());

	// save 
	cout << "Save data.. " << endl;
	data_TMBuD.save_BinaryFile(center_sift, "../../Database/TMBuD/Centroid_SIFT_100.bin");
	data_TMBuD.save_BinaryFile(center_orb, "../../Database/TMBuD/Centroid_ORB_100.bin");
	data_TMBuD.save_BinaryFile(center_hist, "../../Database/TMBuD/Centroid_hist_100.bin");

	// compute hist of visual words 
	for (int i = 0; i < data_TMBuD.getData().size(); i++) {
		Image myImage_ = data_TMBuD.getData()[i];
		data_TMBuD.setCentroids(center_sift);
		Mat histBoW_sift = computeHistogram(myImage_.getSIFT(), data_TMBuD.getCentroids());
		data_TMBuD.setCentroids(center_orb);
		Mat histBoW_orb = computeHistogram(myImage_.getORB(), data_TMBuD.getCentroids());
		data_TMBuD.setCentroids(center_hist);
		Mat histBoW_hist = computeHistogram(myImage_.getHistogram(), data_TMBuD.getCentroids());

		myImage_.setSIFT(histBoW_sift);
		myImage_.setORB(histBoW_orb);
		myImage_.setHistogram(histBoW_hist);

		data_TMBuD.setImageInData(i, myImage_);
		cout << "." << " ";
	}

	data_TMBuD.writeXML(data_TMBuD.getData(), "../../Database/TMBuD/TMBuD.xml", "TMBuD");
}