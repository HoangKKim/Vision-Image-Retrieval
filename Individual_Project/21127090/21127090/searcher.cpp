#include "query_function.h"

void Searcher::setDatabase(string file_name_db, string database_type, string feature_type, string file_name_centroids, string file_name_labels) {
    this->data.loadDatabase(file_name_db, database_type, feature_type);
    this->data.loadCentroids(file_name_centroids);
    this->data.loadLabelsInfo(file_name_labels);
}

vector<float>Searcher::getRecall() {
    return this->recall;
}
void Searcher::addRecall(float recall) {
    this->recall.push_back(recall);
}


vector<float>Searcher::getAveragePrecision() {
    return this->averagePrecision;
}
void Searcher::addAveragePrecision(float averagePrecision) {
    this->averagePrecision.push_back(averagePrecision);
}

vector<Image> Searcher::getResultImages() {
    return this->resultImages;
}
void Searcher::addResultImages(Image image) {
    this->resultImages.push_back(image);
}

bool Searcher::compareBySecond(const pair<int, double>& a, const pair<int, double >& b) {
    return a.second < b.second;
}

vector<pair<int, double>> Searcher::searchBySIFT(Mat queryImage) {
    // process query image
    Mat queryHistogram = processQueryImages(queryImage, "SIFT", this->data.getCentroid());

    // compute distance between query & database
    vector<pair<int, double>> distances;
    for (int i = 0; i < data.getData().size(); i++) {
        Image image = data.getData()[i];
        double dist = computeDistance(queryHistogram, image.getSIFT());
        distances.push_back(make_pair(i, dist));
    }

    // sort 
    std::sort(distances.begin(), distances.end(), compareBySecond);
    return distances;
}

vector<pair<int, double>> Searcher::searchByORB(Mat queryImage) {
    // process query image
    Mat queryHistogram = processQueryImages(queryImage, "ORB", this->data.getCentroid());

    // compute distance between query & database
    vector<pair<int, double>> distances;
    for (int i = 0; i < data.getData().size(); i++) {
        Image image = data.getData()[i];
        double dist = computeDistance(queryHistogram, image.getORB());
        distances.push_back(make_pair(i, dist));
    }

    // sort 
    std::sort(distances.begin(), distances.end(), compareBySecond);
    return distances;
}

vector<pair<int, double>> Searcher::searchByHistogram(Mat queryImage) {
    // process query image
    Mat queryHistogram = processQueryImages(queryImage, "Histogram", this->data.getCentroid());

    // compute distance between query & database
    vector<pair<int, double>> distances;
    for (int i = 0; i < data.getData().size(); i++) {
        Image image = data.getData()[i];
        double dist = computeDistance(queryHistogram, image.getHistogram());
        distances.push_back(make_pair(i, dist));
    }

    // sort 
    std::sort(distances.begin(), distances.end(), compareBySecond);
    return distances;
}

Mat Searcher::displayResultImages(vector<Mat>& vecMat, int windowHeight, int nRows) {
    // Number of images
    int N = vecMat.size();

    // Adjust number of rows to be within the limits of available images
    nRows = nRows > N ? N : nRows;

    // Edge thickness around each image
    int edgeThickness = 10;

    // Number of images per row
    int imagesPerRow = ceil(double(N) / nRows);;

    // Calculate the height for resizing each image
    int resizeHeight = floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness;

    // Maximum length of rows
    int maxRowLength = 0;

    // Vector to store widths after resizing
    vector<int> resizeWidth;

    // Iterate through each row of images
    for (int i = 0; i < N;) {
        int thisRowLen = 0;
        // Process images in current row
        for (int k = 0; k < imagesPerRow; k++) {
            // Calculate aspect ratio of current image
            double aspectRatio = double(vecMat[i].cols) / vecMat[i].rows;
            // Calculate width after resizing based on calculated height
            int temp = int(ceil(resizeHeight * aspectRatio));
            // Store width
            resizeWidth.push_back(temp);
            // Update row length
            thisRowLen += temp;
            // Move to next image
            if (++i == N) break;
        }
        // Update maximum row length if necessary
        if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
            maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
        }
    }

    // Calculate total width of the canvas
    int windowWidth = maxRowLength;

    // Create a black canvas image to place all resized images
    Mat canvasImage(windowHeight, windowWidth, CV_8UC3, Scalar(0, 0, 0));

    // Place each image on the canvas in a grid layout
    for (int k = 0, i = 0; i < nRows; i++) {
        int y = i * resizeHeight + (i + 1) * edgeThickness;
        int x_end = edgeThickness;
        for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
            int x = x_end;
            // Define the region of interest (ROI) for current image
            Rect roi(x, y, resizeWidth[k], resizeHeight);
            Size s = canvasImage(roi).size();
            // Create a target ROI with 3 channels (color image)
            Mat target_ROI(s, CV_8UC3);
            // Convert grayscale images to BGR if needed
            if (vecMat[k].channels() != canvasImage.channels()) {
                if (vecMat[k].channels() == 1) {
                    cvtColor(vecMat[k], target_ROI, COLOR_GRAY2BGR);
                }
            }
            else {
                // Directly copy if already 3 channels
                vecMat[k].copyTo(target_ROI);
            }
            // Resize the image to fit the ROI
            resize(target_ROI, target_ROI, s);
            // Ensure the type of the resized image matches the canvas type
            if (target_ROI.type() != canvasImage.type()) {
                target_ROI.convertTo(target_ROI, canvasImage.type());
            }
            // Copy the resized image to the canvas at the defined ROI
            target_ROI.copyTo(canvasImage(roi));
            // Update x coordinate for the next image
            x_end += resizeWidth[k] + edgeThickness;
        }
    }
    // Return the final canvas image
    return canvasImage;
}

double Searcher::computeAveragePrecision(string queryLabel) {
    int count = 0;

    // count number of image is retrived true
    for (int i = 0; i < this->resultImages.size(); i++) {
        if (queryLabel == resultImages[i].getLabel()) {
            count++;
        }
    }
    this->trueResult = count;

    double result = double((double)count / (double)this->resultImages.size());
    return result;
}

double Searcher::computeRecall(string queryLabel) {
    int count = 0;
    for (int i = 0; i < this->data.getLabelsInfo().size(); i++) {
        if (queryLabel == this->data.getLabelsInfo()[i].first) {
            double result = (double)this->trueResult / (double)this->data.getLabelsInfo()[i].second;
            return result;
        }
    }
}


double Searcher::computeDistance(Mat query, Mat database) {
    double dist = cv::norm(query - database, cv::NORM_L2);
    return dist;
}

Mat Searcher::processQueryImages(Mat image, string feature_type, Mat centers) {
    Mat queryImage;
    // scale image in the same size with database images
    resize(image, queryImage, Size(), 0.5, 0.5); // Resize decrease 50%

    Mat queryFeature;
    Image myImage;
    // extract feature
    if (feature_type == "SIFT" || feature_type == "ORB") {
        queryFeature = myImage.extractLocalFeature(queryImage, feature_type);
    }
    else if (feature_type == "Histogram") {
        queryFeature = myImage.computeColorHistogram(queryImage);
    }

    // compute hist bag of words
    Mat queryHistogram = computeHistogram(queryFeature, centers);
    return queryHistogram;
}

void Searcher::removeResultImages() {
    this->resultImages.clear();
}

void Searcher::setResultImages(int kClosestImages, vector<pair<int, double>> sorted_images) {
    for (int i = 0; i < kClosestImages; i++) {
        Image myImage = data.getData()[sorted_images[i].first];
        addResultImages(myImage);
    }
}

void Searcher::printResultInfor(double AP, double recall, int kImages, double duration) {

    cout << endl << "---------------------------------------------------" << endl;
    cout << "List of result images: " << endl;
    for (int i = 0; i < kImages; i++) {
        Image image = this->resultImages[i];
        cout << image.getFileName() << " " << image.getLabel() << endl;
    }
    cout << endl;
    cout << "- Time for retrival: " << duration << " sec" << endl;
    cout << "- AP: " << AP << endl;
    cout << "- Recall: " << recall << endl;
}

void Searcher::showResult(Mat queryImage, int kImages) {
    int nRow;
    switch (kImages)
    {
    case 3:
        nRow = 2;
        break;
    case 5:
        nRow = 3;
        break;
    case 11:
        nRow = 4;
        break;
    default:
        nRow = 5;
        break;
    }
    vector<Mat> result;
    for (int i = 0; i < kImages; i++) {
        Mat image = imread(this->resultImages[i].getFileName());
        result.push_back(image);
    }
    Mat result_image = displayResultImages(result, 800, nRow);
    resize(queryImage, queryImage, Size(), 0.5, 0.5);
    imshow("query", queryImage);
    imshow("result", result_image);
    waitKey();
    destroyWindow("query");
    destroyWindow("result");
}
