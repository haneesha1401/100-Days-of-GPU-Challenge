#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
    // ----- Step 1: Initialize camera -----
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open camera!" << endl;
        return -1;
    }

    Mat frame, img;
    cout << "Press 'c' to capture the image..." << endl;

    // ----- Step 2: Capture image -----
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        imshow("Camera Feed", frame);
        char key = (char)waitKey(30);
        if (key == 'c') {
            img = frame.clone();
            cout << "✅ Image captured!" << endl;
            break;
        } else if (key == 27) {
            cout << "Exiting..." << endl;
            return 0;
        }
    }

    destroyWindow("Camera Feed");
    cap.release();

    // ----- Step 3: Convert to grayscale -----
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    imshow("Step 1: Original Image", img);
    imshow("Step 2: Grayscale Image", gray);
    waitKey(500);

    // ----- Step 4: Detect keypoints using ORB -----
    Ptr<ORB> orb = ORB::create(500);
    vector<KeyPoint> keypoints;
    Mat descriptors;
    orb->detect(gray, keypoints);

    Mat img_keypoints;
    drawKeypoints(img, keypoints, img_keypoints, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);
    imshow("Step 3: FAST Keypoints (Detected Corners)", img_keypoints);
    waitKey(500);

    // ----- Step 5: Compute descriptors -----
    orb->compute(gray, keypoints, descriptors);

    // ----- Step 6: Visualize orientation and descriptor regions -----
    Mat img_oriented = img.clone();
    for (auto &kp : keypoints) {
        // Draw orientation line (arrow)
        Point2f start = kp.pt;
        float angle = kp.angle * CV_PI / 180.0;
        Point2f end = Point2f(start.x + 15 * cos(angle),
                              start.y + 15 * sin(angle));
        arrowedLine(img_oriented, start, end, Scalar(0, 0, 255), 1, LINE_AA);
    }
    imshow("Step 4: ORB Orientation Visualization", img_oriented);
    waitKey(500);

    // ----- Step 7: Draw keypoints + descriptors as small patches -----
    Mat img_desc = img.clone();
    for (auto &kp : keypoints) {
        circle(img_desc, kp.pt, 8, Scalar(255, 0, 0), 1);
    }
    imshow("Step 5: ORB Descriptor Regions", img_desc);

    cout << "\n✅ ORB feature extraction complete!" << endl;
    cout << "Total Keypoints Detected: " << keypoints.size() << endl;
    cout << "Descriptor Size: " << descriptors.cols << " bytes per keypoint" << endl;

    waitKey(0);
    destroyAllWindows();
    return 0;
}

