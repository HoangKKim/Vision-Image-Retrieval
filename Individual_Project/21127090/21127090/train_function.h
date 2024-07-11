#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

using namespace std;
using namespace cv;

class Image {
private:
    string filename;
    string label;
    Mat sift;
    Mat orb;
    Mat histogram;
    Mat correlogram;
public:
    // get & set
    string getFileName();
    string getLabel();
    Mat getSIFT();
    Mat getORB();
    Mat getHistogram();
    Mat getCorrelogram();

    void setFileName(string filename);
    void setLabel(string label);
    void setSIFT(Mat sift);
    void setORB(Mat orb);
    void setHistogram(Mat histogram);
    void setCorrelogram(Mat correlogram);

    Mat extractLocalFeature(Mat image, string type);
    Mat computeColorHistogram(Mat image);
    Mat computeCorrelogram(const Mat& image);
};

class Dataset {
private:
    vector<Image>  data;
    Mat centroids;
public:
    vector<Image> getData();
    void setData(vector<Image>data);

    void setImageInData(int index, Image image) {
        this->data[index] = image;
    }

    Mat getCentroids();
    void setCentroids(Mat centers);

    vector<Image> readCSV(const string& folder, const string filename_csv);
    string getFileName(const string& path);
    vector<Image> createLabel(const string& folder);
    vector<Image> readXML(string fileName_XML, string type);
    Mat load_BinaryFile(const string& filename);

    void writeXML(vector <Image> data, string filenName_XML, string type);
    void save_BinaryFile(Mat descriptor, string fileName);

};

//class Trainer {
//private:
//    Dataset database;
//public:
//    Dataset getDatabase();
//    Mat buildVisualWords(Mat const& allDescriptor, int nClusters);
//    Mat computeHistogram(const Mat& descriptors, const Mat& centers);
//};


//Dataset getDatabase();
Mat buildVisualWords(Mat const& allDescriptor, int nClusters, Mat previousCenter);
Mat computeHistogram(const Mat& descriptors, const Mat& centers);
void preprocess_data();




