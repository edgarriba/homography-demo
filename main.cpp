#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

const float nn_match_ratio = 0.8f;

void compute_and_draw_homography(const std::vector<DMatch>& good_matches,
                                 const std::vector<KeyPoint>& kpts1,
                                 const std::vector<KeyPoint>& kpts2,
                                 const cv::Mat& img_object,
                                 cv::Mat& img_out,
                                 const int id);

void ratio_test(const std::vector<std::vector<DMatch> >& nn_matches,
                std::vector<DMatch>& good_matches,
                const float nn_match_ratio);

int main( int argc, char** argv ) {
    if (argc < 2) {
        std::cout << "Usage: ./main [<reference_image>]" << std::endl;
        return -1; 
    }

    Mat img_object = imread( argv[1], IMREAD_GRAYSCALE);

    if (!img_object.data) {
        std::cout<< "[ERROR]: Cannot read reference image" << std::endl;
        return -1;
    }

    // initialize keypoins detectors and matcher
    Ptr<SURF>  detector1 = SURF::create(400);
    Ptr<AKAZE> detector2 = AKAZE::create();

    BFMatcher matcher;

    // create local descriptors detector
    // TODO(edgar): add tiny-dnn based one
  
    Mat desc11, desc12, desc21, desc22;
    std::vector<KeyPoint> kpts11, kpts12, kpts21, kpts22;

    // detect keypoints and descriptors from reference image
    detector1->detectAndCompute(img_object, Mat(), kpts11, desc11);
    detector2->detectAndCompute(img_object, Mat(), kpts21, desc21);

    // initialize video caption and visualization stuff

    VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout  << "[ERROR]: Could not open video capture" << std::endl;
        return -1;
    }

    namedWindow("Main window", WINDOW_AUTOSIZE);

    // video loop
    
    Mat frame;

    for (;;) {

        cap >> frame;

        if (frame.empty()) {
            std::cout << "[ERROR] Cannot capture frame" << std::endl;
            continue;
        }
        
        // detect scene keypoints and local descriptors
        detector1->detectAndCompute(frame, Mat(), kpts12, desc12);
        detector2->detectAndCompute(frame, Mat(), kpts22, desc22);

        // match descriptors vectors
        std::vector<std::vector<DMatch> > nn_matches1, nn_matches2;
        matcher.knnMatch(desc11, desc12, nn_matches1, 2);
        matcher.knnMatch(desc21, desc22, nn_matches2, 2);

        // remove some early detected outliers
        std::vector<cv::DMatch> good_matches1, good_matches2;
        ratio_test(nn_matches1, good_matches1, nn_match_ratio);
        ratio_test(nn_matches2, good_matches2, nn_match_ratio);

        // compute the homography ans draw it to the current frame

        if (good_matches1.size() > 0) {
            compute_and_draw_homography(
                good_matches1, kpts11, kpts12, img_object, frame, 0);
        }
        
        if (good_matches2.size() > 0) {
            compute_and_draw_homography(
                good_matches2, kpts21, kpts22, img_object, frame, 1);
        }

        // show detected objects 
        imshow("Main window", frame);
        
        // stop video loop
        char c = static_cast<char>(cv::waitKey(3));
        if (c == 27) break;
    }

    return 0;
}

void compute_and_draw_homography(const std::vector<DMatch>& good_matches,
                                 const std::vector<KeyPoint>& kpts1,
                                 const std::vector<KeyPoint>& kpts2,
                                 const cv::Mat& img_object,
                                 cv::Mat& frame,
                                 int id) {
    // localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for (size_t i = 0; i < good_matches.size(); i++) {
        // get the keypoints from the good matches
        obj.push_back(kpts1[good_matches[i].queryIdx].pt);
        scene.push_back(kpts2[good_matches[i].trainIdx].pt);
    }

    // comptute the homography fitting RANSAC model
    Mat H = findHomography(obj, scene, RANSAC);

    if (H.empty()) {
        std::cout << "[ERROR] Cannot compute homography" << std::endl;
        return;
    }

    // get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0,0);
    obj_corners[1] = Point(img_object.cols, 0);
    obj_corners[2] = Point(img_object.cols, img_object.rows);
    obj_corners[3] = Point(0, img_object.rows);

    // apply the homography to the object points
    std::vector<Point2f> scene_corners(4);
    cv::perspectiveTransform(obj_corners, scene_corners, H);

    // drawing stuff

    Scalar color;
    std::string name;
    cv::Point origin;
    int thickness;

    if (id == 0) {
        color  = cv::Scalar(0, 255, 0);
        name   = std::string("SURF");
        origin = cv::Point(frame.cols - 150, 50);
        thickness = 1;
    } else {
        color  = Scalar(255, 0, 0);
        name   = std::string("AKAZE");
        origin = cv::Point(frame.cols - 150, 90);
        thickness = 4;
    }

    line(frame, scene_corners[0], scene_corners[1], color, thickness);
    line(frame, scene_corners[1], scene_corners[2], color, thickness);
    line(frame, scene_corners[2], scene_corners[3], color, thickness);
    line(frame, scene_corners[3], scene_corners[0], color, thickness);
    
    cv::putText(frame, name, origin, FONT_HERSHEY_SIMPLEX, 1.0, color, 3);
}

void ratio_test(const std::vector<std::vector<DMatch> >& nn_matches,
                std::vector<DMatch>& good_matches,
                const float nn_match_ratio) {
    for (size_t i = 0; i < nn_matches.size(); ++i) {
        DMatch first = nn_matches[i][0];
        float dist1  = nn_matches[i][0].distance;
        float dist2  = nn_matches[i][1].distance;
        if (dist1 < nn_match_ratio * dist2) {
            good_matches.push_back(first);
        }
    }
}



