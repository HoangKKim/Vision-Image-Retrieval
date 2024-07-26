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

Database Searcher::getData() {
    return this->data;
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
        //double dist = computeDistance(queryHistogram, image.getHistogram());
        double dist = compareHist(queryHistogram, image.getHistogram(), HISTCMP_CORREL);
        distances.push_back(make_pair(i, dist));
    }

    // sort 
    std::sort(distances.begin(), distances.end(),
        [](const pair<int, double>& a, const pair<int, double >& b) {
            return a.second > b.second;
        });
    return distances;
}

vector<pair<int, double>> Searcher::searchByCorrelogram(Mat queryImage) {
    // process query image
    Mat queryHistogram = processQueryImages(queryImage, "Correlogram", this->data.getCentroid());

    // compute distance between query & database
    vector<pair<int, double>> distances;
    for (int i = 0; i < data.getData().size(); i++) {
        Image image = data.getData()[i];
        //double dist = norm(queryHistogram, image.getCorrelogram(), NORM_L2);
        double dist = compareHist(queryHistogram, image.getCorrelogram(), NORM_L2);
        distances.push_back(make_pair(i, dist));
    }

    // sort 
    std::sort(distances.begin(), distances.end(), compareBySecond);
    return distances;
}

vector<pair<int, double>> Searcher::searchByCombine(Mat queryImage) {
    // process query image
    Mat queryHistogram = processQueryImages(queryImage, "Combine", this->data.getCentroid());

    // compute distance between query & database
    vector<pair<int, double>> distances;
    for (int i = 0; i < data.getData().size(); i++) {
        Image image = data.getData()[i];
        double dist = computeDistance(queryHistogram, image.getCombine());
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

double Searcher::computeDistance(Mat query, Mat database) {
    //double dist = cv::norm(query - database, cv::NORM_L2);
    vector<DMatch> matches;
    FlannBasedMatcher matcher;
    matcher.match(query, database, matches);
    float dist = 0.0;
    for (const auto& match : matches) {
        dist += match.distance;
    }
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
        return queryFeature;
    }
    else if (feature_type == "Correlogram") {
        queryFeature = myImage.computeCorrelogram(queryImage);
        return queryFeature;
    }
    // compute hist bag of words
    Mat queryHistogram = computeHistogram(queryFeature, centers);
    return queryHistogram;
}

void Searcher::removeResultImages() {
    this->resultImages.clear();
}

void Searcher::setResultImages(int kClosestImages, vector<pair<int, double>> sorted_images, string data_type) {
    if (data_type == "CD") {
        for (int i = 0; i < kClosestImages; i++) {
            // get info of k-closest images and add into class's attribute
            Image myImage = data.getData()[sorted_images[i].first];
            addResultImages(myImage);
        }
    }
    else {
        for (int i = 1; i <= kClosestImages; i++) {
            // get info of k-closest images and add into class's attribute
            Image myImage = data.getData()[sorted_images[i].first];
            addResultImages(myImage);
        }
    }
}

void Searcher::printResultInfor(double AP, double recall, int kImages, double duration, string queryImageName, string label) {

    cout << endl << "---------------------------------------------------" << endl;
    cout << "List of result for query image: " << queryImageName<<" - Label: "<<label<< endl;
    for (int i = 0; i < kImages; i++) {
        Image image = this->resultImages[i];
        cout << image.getFileName() << " " << image.getLabel() << endl;
    }
    cout << endl;
    cout << "- Time for retrival: " << duration << " seconds" << endl;
    cout << "- Recall: " << recall << endl;
    cout << "- AP: " << AP << endl;
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


vector<double> Searcher::computePrecision(string queryLabel) {
    int count = 0;
    std::vector<double> precision_at_k;

    // count number of images retrieved correctly
    for (int i = 0; i < this->resultImages.size(); i++) {
        if (queryLabel == resultImages[i].getLabel()) {
            count++;
            double precision = double(count) / double(i + 1);
            precision_at_k.push_back(precision);
        }
    }
    this->trueResult = count;
    return precision_at_k;
}

double Searcher::computeRecall(string queryLabel) {
    int count = 0;
    for (int i = 0; i < this->data.getLabelsInfo().size(); i++) {
        if (queryLabel == this->data.getLabelsInfo()[i].first) {
            double result = (double)this->trueResult / (double)this->data.getLabelsInfo()[i].second;
            return result;
        }
    }
    return 0.0;
}

double Searcher::computeAveragePrecision(string queryLabel) {
    std::vector<double> precision_at_k = computePrecision(queryLabel);

    if (precision_at_k.empty()) {
        return 0.0; // no image is retrived correctly
    }

    double sum_precision = 0.0;
    for (double precision : precision_at_k) {
        sum_precision += precision;
    }

    double average_precision = sum_precision / precision_at_k.size();
    return average_precision;
}
