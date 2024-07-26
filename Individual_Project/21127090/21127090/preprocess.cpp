#include "train_function.h"


void preprocess_data() {
	string csvFileName = "DATASET\ SPLIT.csv";
	string folder_TMBuD = "../../Data/TMBuD-main/TMBuD-main";
	string folder_CD = "../../Data/CD_data/training_set/training_images";

	Dataset data_CD, data_TMBuD;

	//process the initial data
	data_TMBuD.readCSV(folder_TMBuD, csvFileName);
	data_CD.createLabel(folder_CD);

	// write the dataset in the new type of file (xml)
	data_TMBuD.writeXML(data_TMBuD.getData(), "../../Database/TMBuD/TMBuD.xml", "TMBuD");
	data_CD.writeXML(data_CD.getData(), "../../Database/CD/CD.xml", "CD");
	
	// count number of each label in a dataset
	data_CD.countLabels("../../Database/CD/CD.xml", "CD", "../../Database/CD/labels.txt");
	data_TMBuD.countLabels("../../Database/TMBuD/TMBuD.xml", "TMBuD", "../../Database/TMBuD/labels.txt");

	// extract 
	//process data in CD file
	Mat sift_descriptors, orb_descriptors, color_hist_descriptors;
	// extract feature
	for (int i = 0; i < data_CD.getData().size(); i++) {
		Image myImage = data_CD.getData()[i];
		// read image
		Mat image = imread(myImage.getFileName());
		// resize image
		resize(image, image, Size(), 0.5, 0.5); // Resize decrease 50%
		vector<KeyPoint> key;
		Mat sift = myImage.extractLocalFeature(image, "SIFT");
		Mat orb = myImage.extractLocalFeature(image, "ORB");
		myImage.setSIFT(sift);
		myImage.setORB(orb);

		data_CD.setImageInData(i, myImage);
		sift_descriptors.push_back(sift);
		orb_descriptors.push_back(orb);
		cout << i + 1 << "..." << " ";
	}

	// find visual words
	cout << "\nKmeans..." << endl;
	Mat center_sift = buildVisualWords(sift_descriptors, 100);
	Mat center_orb = buildVisualWords(orb_descriptors, 100);

	// save 
	cout << "Save data.. " << endl;
	data_CD.save_BinaryFile(center_sift, "../../Database/CD/Centroid_SIFT_100.bin");
	data_CD.save_BinaryFile(center_orb, "../../Database/CD/Centroid_ORB_100.bin");

	// compute hist of visual words 
	for (int i = 0; i < data_CD.getData().size(); i++) {
		Image myImage_ = data_CD.getData()[i];
		data_CD.setCentroids(center_sift);
		Mat histBoW_sift = computeHistogram(myImage_.getSIFT(), data_CD.getCentroids());
		data_CD.setCentroids(center_orb);
		Mat histBoW_orb = computeHistogram(myImage_.getORB(), data_CD.getCentroids());

		myImage_.setSIFT(histBoW_sift);
		myImage_.setORB(histBoW_orb);

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

		vector<KeyPoint> key;
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
	center_sift = buildVisualWords(sift_descriptors, 100);
	center_orb = buildVisualWords(orb_descriptors, 100);

	// save 
	cout << "Save data.. " << endl;
	data_TMBuD.save_BinaryFile(center_sift, "../../Database/TMBuD/Centroid_SIFT_100.bin");
	data_TMBuD.save_BinaryFile(center_orb, "../../Database/TMBuD/Centroid_ORB_100.bin");

	// compute hist of visual words 
	for (int i = 0; i < data_TMBuD.getData().size(); i++) {
		Image myImage_ = data_TMBuD.getData()[i];
		data_TMBuD.setCentroids(center_sift);
		Mat histBoW_sift = computeHistogram(myImage_.getSIFT(), data_TMBuD.getCentroids());
		data_TMBuD.setCentroids(center_orb);
		Mat histBoW_orb = computeHistogram(myImage_.getORB(), data_TMBuD.getCentroids());

		myImage_.setSIFT(histBoW_sift);
		myImage_.setORB(histBoW_orb);

		data_TMBuD.setImageInData(i, myImage_);
		cout << "." << " ";
	}

	data_TMBuD.writeXML(data_TMBuD.getData(), "../../Database/TMBuD/TMBuD.xml", "TMBuD");
}

// tach cac feature ra thanh local va global
void preprocess_data_localFeature() {
	// read dataset from xml file
	Dataset myDataset;
	//myDataset.readXML("../../Database/TMBuD/TMBuD.xml", "TMBuD");
	//string localFeature_folder = "../../Database/TMBuD/";
	myDataset.readXML("../../Database/CD/CD.xml", "CD");
	string localFeature_folder = "../../Database/CD/";
	vector <Mat> histgrams;

	cout << "extract feature" << endl;
	for (int i = 0; i < myDataset.getData().size(); i++) {
		Image myImage = myDataset.getData()[i];
		myImage.setHistogram(Mat());
		myImage.setCorrelogram(Mat());

		myDataset.setImageInData(i, myImage);
		cout << i + 1 << "..";
	}
	//myDataset.writeXML(myDataset.getData(), localFeature_folder + "TMBuD_LocalFeature.xml", "TMBuD");
	myDataset.writeXML(myDataset.getData(), localFeature_folder + "CD_LocalFeature.xml", "CD");

}


void preprocess_data_histogram() {
	// read dataset from xml file
	Dataset myDataset;
	myDataset.readXML("../../Database/CD/CD.xml", "CD");
	
	string Histogram_folder = "../../Database/CD/";
	cout << "extract feature" << endl;
	for (int i = 0; i < myDataset.getData().size(); i++) {
		Image myImage = myDataset.getData()[i];
		Mat image = imread(myImage.getFileName());
		resize(image, image, Size(), 0.5, 0.5); // Resize decrease 50%

		// compute histogram
		Mat histogram = myImage.computeColorHistogram(image);
		myImage.setHistogram(histogram);
		myImage.setCorrelogram(Mat());
		myImage.setSIFT(Mat());
		myImage.setORB(Mat());


		myDataset.setImageInData(i, myImage);
		cout << i + 1 << "..";
	}
	myDataset.writeXML(myDataset.getData(), Histogram_folder + "CD_Histogram.xml", "CD");
}

void preprocess_data_correlogram() {
	// read dataset from xml file
	Dataset myDataset;
	myDataset.readXML("../../Database/CD/CD.xml", "CD");

	string Correlogram_folder = "../../Database/CD/";
	cout << "extract feature" << endl;
	vector<Mat> Corre;
	for (int i = 0; i < myDataset.getData().size(); i++) {
		Image myImage = myDataset.getData()[i];
		Mat image = imread(myImage.getFileName());
		resize(image, image, Size(), 0.5, 0.5); // Resize decrease 50%
		
		//compute histogram
		Mat correlogram = myImage.computeCorrelogram(image);
		
		myImage.setCorrelogram(correlogram);

		myImage.setHistogram(Mat());
		myImage.setSIFT(Mat());
		myImage.setORB(Mat());

		myDataset.setImageInData(i, myImage);
		cout << i + 1 << "..";
		Corre.push_back(correlogram);
	}
	myDataset.save_BinaryFile(Corre, Correlogram_folder + "all_corre.bin");
	myDataset.writeXML(myDataset.getData(), Correlogram_folder + "CD_Correlogram.xml", "CD");
}