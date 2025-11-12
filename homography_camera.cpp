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

    Mat frame, img1, img2;
    cout << "Press 'c' to capture the first image..." << endl;

    // ----- Step 2: Capture first image -----
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        imshow("Camera Feed", frame);

        char key = (char)waitKey(30);
        if (key == 'c') {
            img1 = frame.clone();
            cout << "✅ First image captured!" << endl;
            break;
        }
        else if (key == 27) { // ESC key
            cout << "Exiting..." << endl;
            return 0;
        }
    }

    cout << "Press 'c' again to capture the second image..." << endl;

    // ----- Step 3: Capture second image -----
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        imshow("Camera Feed", frame);

        char key = (char)waitKey(30);
        if (key == 'c') {
            img2 = frame.clone();
            cout << "✅ Second image captured!" << endl;
            break;
        }
        else if (key == 27) {
            cout << "Exiting..." << endl;
            return 0;
        }
    }

    destroyWindow("Camera Feed");
    cap.release();

    // ----- Step 4: Detect ORB features -----
    Ptr<ORB> orb = ORB::create(5000);
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    orb->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    // ----- Step 5: Match features using BFMatcher -----
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Sort matches by distance (best first)
    sort(matches.begin(), matches.end(), [](const DMatch &a, const DMatch &b) {
        return a.distance < b.distance;
    });

    const int numGoodMatches = 50;
    matches.resize(min(numGoodMatches, (int)matches.size()));

    // ----- Step 6: Extract matched points -----
    vector<Point2f> pts1, pts2;
    for (auto &m : matches) {
        pts1.push_back(keypoints1[m.queryIdx].pt);
        pts2.push_back(keypoints2[m.trainIdx].pt);
    }

    // ----- Step 7: Compute Homography -----
    Mat H = findHomography(pts1, pts2, RANSAC);
    if (H.empty()) {
        cerr << "Homography computation failed!" << endl;
        return -1;
    }

    cout << "\nEstimated Homography Matrix:\n" << H << endl;

    // ----- Step 8: Warp first image -----
    Mat warped;
    warpPerspective(img1, warped, H, img2.size());

    // ----- Step 9: Display Results -----
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches,
                Scalar::all(-1), Scalar::all(-1), vector<char>(),
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imshow("Matched Features", img_matches);
    imshow("Warped Image 1", warped);
    imshow("Second Image", img2);

    cout << "\nPress any key to exit..." << endl;
    waitKey(0);
    destroyAllWindows();

    return 0;
}

