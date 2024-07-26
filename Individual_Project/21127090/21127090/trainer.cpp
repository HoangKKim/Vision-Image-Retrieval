#include "train_function.h"

Mat buildVisualWords(Mat const& allDescriptor, int nClusters) {
    Mat labels, centers;
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 20, 0.1);
    int attempts = 5;
    //int flag = cv::KMEANS_PP_CENTERS;
    int flag = cv::KMEANS_USE_INITIAL_LABELS;
    kmeans(allDescriptor, nClusters, labels, criteria, attempts, flag, centers);
    return centers;
}


Mat computeHistogram(const Mat& descriptors, const Mat& centers) {
    int numClusters = centers.rows;     // n words
    Mat histogram = Mat::zeros(1, numClusters, CV_32F);

    for (int i = 0; i < descriptors.rows; i++) {
        Mat descriptor = descriptors.row(i);

        // find the closest cluster
        double minDist = DBL_MAX;
        int bestCluster = -1;
        for (int j = 0; j < centers.rows; j++) {
            // using euclidean
            Mat temp = centers.row(j);
            double dist = norm(descriptor - centers.row(j), NORM_L2);
            if (dist < minDist) {
                minDist = dist;
                bestCluster = j;
            }
        }

        // increase value for the corresponding bin 
        histogram.at<float>(0, bestCluster)++;
    }
    return histogram;
}