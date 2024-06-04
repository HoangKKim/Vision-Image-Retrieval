#include "function.h"

int main(int n_args, char* args[]) {

    if (n_args !=2) {
        cout << "Invalid action!";
        return 0;
    }
    String inputReq = args[1];
    String video_filePath;
    bool usingWebcam = false;


    if (inputReq == "0") {
        usingWebcam = true;
    }
    else
        video_filePath = inputReq;
    Mat frame;
    // constructor
    Video myVideo(video_filePath, usingWebcam);

    // whether video could be opened, if not, using webcam alternative
    if (usingWebcam == false) {
        myVideo.openVideo();
    }
    else myVideo.openWebcam();

    while (true) {
        // get a frame from the video
        frame = myVideo.getFrame();

        // check the empty frame (if is empty, the video is end)
        if (frame.empty()) {
            break;
        }

        // display the frame and its histogram
        Mat histImg = myVideo.drawHistogram_RGB(frame, 512, 400);
        myVideo.showResult(frame, histImg);
        imwrite("frame_image.png", frame);
        imwrite("hist_image.png", histImg);


        // Set hot key p to break the loop and set the time pausing for each frame
        if (waitKey(500) == 'q') {
            break;
        }
    }

    // release the video
    myVideo.~Video();
    destroyAllWindows();

    return 0;
}
