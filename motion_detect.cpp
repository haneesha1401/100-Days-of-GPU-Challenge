#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // -------------------------------
    // Task 1: Setup and Initialization
    // -------------------------------
    VideoCapture cap(0); // 0 for webcam, or replace with "video.mp4" for file
    if (!cap.isOpened()) {
        cout << "Error: Could not open video source!" << endl;
        return -1;
    }

    // Get frame properties
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    cout << "Frame size: " << frame_width << "x" << frame_height << endl;

    // Optional: to save output
    VideoWriter output("motion_output.avi",
                       VideoWriter::fourcc('M','J','P','G'),
                       20, Size(frame_width, frame_height));

    // -------------------------------
    // Task 2: Capture Frames and Perform Frame Differencing
    // -------------------------------
    Mat frame, gray, prevGray, diff, thresh;
    bool firstFrame = true;

    while (true) {
        bool ret = cap.read(frame);
        if (!ret) {
            cout << "End of video or error reading frame." << endl;
            break;
        }

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(5, 5), 0);

        if (firstFrame) {
            gray.copyTo(prevGray);
            firstFrame = false;
            continue;
        }

        // Absolute difference between current and previous frames
        absdiff(gray, prevGray, diff);

        // Threshold to highlight motion areas
        threshold(diff, thresh, 25, 255, THRESH_BINARY);

        // Dilation to remove noise
        dilate(thresh, thresh, Mat(), Point(-1, -1), 2);

        // -------------------------------
        // Task 3: Detect Contours and Track Motion
        // -------------------------------
        vector<vector<Point>> contours;
        findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (auto &contour : contours) {
            if (contourArea(contour) < 500) // filter small areas
                continue;

            Rect box = boundingRect(contour);
            rectangle(frame, box, Scalar(0, 255, 0), 2);
        }

        // -------------------------------
        // Task 4: Display and Save Output
        // -------------------------------
        imshow("Motion Detection", frame);
        output.write(frame); // optional

        // Press ESC to exit
        if (waitKey(30) == 27)
            break;

        gray.copyTo(prevGray);
    }

    cap.release();
    output.release();
    destroyAllWindows();

    return 0;
}

