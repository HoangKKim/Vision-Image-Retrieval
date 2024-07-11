#pragma once
#include "train_function.h"

class Database {
private:
	vector<Image> data;
	Mat centers;
	vector<pair<string, int>> labels_info;
public:
	vector<Image> getData();
	Mat getCentroid();
	vector<pair<string, int>> getLabelsInfo();
	void loadCentroids(string file_name);
	void loadDatabase(string file_name, string database_type, string feature_type);
	void loadLabelsInfo(string file_name);
};

class Searcher {
private:
	Database data;
	vector<float> averagePrecision;
	vector<float> recall;
	vector<Image> resultImages;
	int trueResult;
public:
	vector<float> getRecall();
	void addRecall(float recall);

	vector<float> getAveragePrecision();
	void addAveragePrecision(float averagePrecision);

	vector<Image> getResultImages();
	void addResultImages(Image image);

	void setDatabase(string file_name_db, string database_type, string feature_type, string file_name_centroids, string file_name_labels);	Database getData();

	double computeDistance(Mat query, Mat database);
	Mat processQueryImages(Mat image, string feature_type, Mat centers);


	vector<pair<int, double>> searchBySIFT(Mat queryImage);
	vector<pair<int, double>> searchByORB(Mat queryImage);
	vector<pair<int, double>> searchByHistogram(Mat queryImage);

	static bool compareBySecond(const pair<int, double>& a, const pair<int, double>& b);

	void removeResultImages();

	void setResultImages(int kClosestImages, vector<pair<int, double>> sorted_images);

	Mat displayResultImages(vector<Mat>& vecMat, int windowHeight, int nRows);
	double computeAveragePrecision(string queryLabel);
	double computeRecall(string queryLabel);

	void printResultInfor(double AP, double recall, int kImages, double duration);
	void showResult(Mat queryImage, int kImages);
};


class Query {
private:
	vector<pair<string, string>> images;
public:
	vector<pair<string, string>> getImages();
	void loadQueryImages(string file_name);
};