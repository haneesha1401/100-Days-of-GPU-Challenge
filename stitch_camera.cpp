// stitch_camera_visual.cpp
// Compile with:
// g++ stitch_camera_visual.cpp -o stitch_camera_visual `pkg-config --cflags --libs opencv4`

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "❌ Cannot open camera!" << endl;
        return -1;
    }

    Mat img1, img2;
    cout << "📸 Capture two images for stitching.\nPress SPACE to capture, ESC to quit.\n";

    // --- Capture first image ---
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) continue;
        imshow("Camera - Capture 1", frame);
        int key = waitKey(30);
        if (key == 27) return 0;
        if (key == ' ') { img1 = frame.clone(); break; }
    }
    destroyWindow("Camera - Capture 1");

    // --- Capture second image ---
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) continue;
        imshow("Camera - Capture 2", frame);
        int key = waitKey(30);
        if (key == 27) return 0;
        if (key == ' ') { img2 = frame.clone(); break; }
    }
    destroyWindow("Camera - Capture 2");
    cap.release();

    // --- Convert to grayscale ---
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    // --- ORB Feature Detection ---
    Ptr<ORB> orb = ORB::create(5000);
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    orb->detectAndCompute(gray1, noArray(), kpts1, desc1);
    orb->detectAndCompute(gray2, noArray(), kpts2, desc2);

    // --- Visualize keypoints ---
    Mat kp_img1, kp_img2;
    drawKeypoints(img1, kpts1, kp_img1, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(img2, kpts2, kp_img2, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("ORB Keypoints Image 1", kp_img1);
    imshow("ORB Keypoints Image 2", kp_img2);
    waitKey(0);

    // --- BFMatcher ---
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(desc1, desc2, matches);

    // --- Filter good matches ---
    double min_dist = 100;
    for (auto& m : matches)
        if (m.distance < min_dist) min_dist = m.distance;

    vector<DMatch> good_matches;
    for (auto& m : matches)
        if (m.distance <= max(2*min_dist, 30.0))
            good_matches.push_back(m);

    // --- Visualize matches ---
    Mat match_img;
    drawMatches(img1, kpts1, img2, kpts2, good_matches, match_img,
                Scalar::all(-1), Scalar::all(-1), vector<char>(),
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Good Matches", match_img);
    waitKey(0);

    // --- Extract matched points ---
    vector<Point2f> pts1, pts2;
    for (auto& m : good_matches) {
        pts1.push_back(kpts1[m.queryIdx].pt);
        pts2.push_back(kpts2[m.trainIdx].pt);
    }

    // --- Find homography ---
    Mat H = findHomography(pts2, pts1, RANSAC);
    if (H.empty()) {
        cerr << "❌ Homography estimation failed!" << endl;
        return -1;
    }

    // --- Warp second image and stitch ---
    Mat result;
    warpPerspective(img2, result, H, Size(img1.cols + img2.cols, max(img1.rows, img2.rows)));
    Mat half(result, Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(half);

    // --- Show and save panorama ---
    imshow("Panorama", result);
    imwrite("panorama_camera_visual.jpg", result);
    cout << "✅ Panorama saved as 'panorama_camera_visual.jpg'\n";

    waitKey(0);
    destroyAllWindows();
    return 0;
}

