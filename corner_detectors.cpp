#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) { cerr << "Cannot open camera\n"; return -1; }

    Mat frame;
    cout << "Press SPACE to capture, ESC to exit.\n";
    while (true) {
        cap >> frame;
        if (frame.empty()) continue;
        imshow("Camera", frame);
        int k = waitKey(30);
        if (k == 27) return 0;
        if (k == ' ') break;
    }
    destroyWindow("Camera");

    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // Harris (example)
    Mat dst, dst_norm;
    cornerHarris(gray, dst, 2, 3, 0.04);
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1);

    Mat harris_img = frame.clone();
    for (int y = 0; y < dst_norm.rows; y++)
        for (int x = 0; x < dst_norm.cols; x++)
            if ((int)dst_norm.at<float>(y, x) > 150)
                circle(harris_img, Point(x, y), 4, Scalar(0, 0, 255), 1);

    // Shi-Tomasi
    vector<Point2f> corners;
    goodFeaturesToTrack(gray, corners, 200, 0.01, 10);
    Mat shi_img = frame.clone();
    for (auto &p : corners) circle(shi_img, p, 4, Scalar(0, 0, 255), 1);

    // FAST
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(25, true);
    vector<KeyPoint> keypoints;
    fast->detect(gray, keypoints);
    Mat fast_img = frame.clone(); // ensure output is valid
    drawKeypoints(frame, keypoints, fast_img, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    imshow("Harris Corners", harris_img);
    imshow("Shi-Tomasi Corners", shi_img);
    imshow("FAST Keypoints", fast_img);

    cout << "Press any key to exit.\n";
    waitKey(0);
    destroyAllWindows();
    return 0;
}

