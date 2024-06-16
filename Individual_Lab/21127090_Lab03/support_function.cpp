#include "function.h"

Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows) {
    // Number of images
    int N = vecMat.size();

    // Adjust number of rows to be within the limits of available images
    nRows = nRows > N ? N : nRows;

    // Edge thickness around each image
    int edgeThickness = 10;

    // Number of images per row
    int imagesPerRow = ceil(double(N) / nRows);

    // Calculate the height for resizing each image
    int resizeHeight = floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness;

    // Maximum length of rows
    int maxRowLength = 0;

    // Vector to store widths after resizing
    std::vector<int> resizeWidth;

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
    cv::Mat canvasImage(windowHeight, windowWidth, CV_8UC3, Scalar(0, 0, 0));

    // Place each image on the canvas in a grid layout
    for (int k = 0, i = 0; i < nRows; i++) {
        int y = i * resizeHeight + (i + 1) * edgeThickness;
        int x_end = edgeThickness;
        for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
            int x = x_end;
            // Define the region of interest (ROI) for current image
            cv::Rect roi(x, y, resizeWidth[k], resizeHeight);
            cv::Size s = canvasImage(roi).size();
            // Create a target ROI with 3 channels (color image)
            cv::Mat target_ROI(s, CV_8UC3);
            // Convert grayscale images to BGR if needed
            if (vecMat[k].channels() != canvasImage.channels()) {
                if (vecMat[k].channels() == 1) {
                    cv::cvtColor(vecMat[k], target_ROI, COLOR_GRAY2BGR);
                }
            }
            else {
                // Directly copy if already 3 channels
                vecMat[k].copyTo(target_ROI);
            }
            // Resize the image to fit the ROI
            cv::resize(target_ROI, target_ROI, s);
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
