/**
 * @function moments_demo.cpp
 * @brief Demo code to calculate moments
 * @author OpenCV team
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

Mat src_gray;
int thresh = 50;
RNG rng(12345);

/// Function header
void thresh_callback(int, void *);

/**
 * @function main
 */
int main(int argc, char **argv)
{
    /// Load source image
    CommandLineParser parser(argc, argv, "{@input | ../data/stuff.jpg | input image}");
    src_gray = imread(parser.get<String>("@input"), IMREAD_GRAYSCALE);

    if (src_gray.empty())
    {
        cout << "Could not open or find the image!\n"
             << endl;
        cout << "usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    /// Convert image to gray and blur it
    // cvtColor( src_gray, src_gray, COLOR_BGR2GRAY );
    // blur( src_gray, src_gray, Size(3,3) );

    /// Create Window
    const char *source_window = "Source";
    namedWindow(source_window);
    imshow(source_window, src_gray);

    const int max_thresh = 255;
    createTrackbar("Canny thresh:", source_window, &thresh, max_thresh, thresh_callback);
    thresh_callback(0, 0);
    waitKey();

    // // convert grayscale to binary image
    // Mat thr;
    // threshold( src_gray, thr, 50, 255, THRESH_BINARY );
    // imshow("Thrsd", thr);

    // Mat labels, stats, centroids;
    // int num_components = connectedComponentsWithStats(thr, labels, stats, centroids);
    // cout << "#Dots: " << num_components << endl;

    // Mat drawing = Mat::zeros( src_gray.size(), CV_8UC3 );
    // for ( int i = 0; i < num_components; i++ )
    // {
    //     Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256) );
    //     // drawContours( drawing, contours, (int)i, color, 1 );
    //     float x = float(centroids.at<double>(i, 1)), y = float(centroids.at<double>(i, 0));
    //     Point2f pt(x,y);
    //     circle( drawing, pt, 1, color, -1 );
    // }

    // // show the image with a point mark at the centroid
    // imshow("Image with center", drawing);
    // waitKey(0);

    return 0;
}

/**
 * @function thresh_callback
 */
void thresh_callback(int, void *)
{
    /// Detect edges using canny
    Mat canny_output;
    Canny(src_gray, canny_output, thresh, thresh * 3, 3);
    imshow("Canny", canny_output);

    /// Find contours
    vector<vector<Point>> contours;
    findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    /// Get the moments
    vector<Moments> mu(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        mu[i] = moments(contours[i]);
    }

    ///  Get the mass centers
    vector<Point2f> mc(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        //add 1e-5 to avoid division by zero
        mc[i] = Point2f(static_cast<float>(mu[i].m10 / (mu[i].m00 + 1e-5)),
                        static_cast<float>(mu[i].m01 / (mu[i].m00 + 1e-5)));
        cout << "mc[" << i << "]=" << mc[i] << endl;
    }

    /// Draw contours
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        // drawContours( drawing, contours, (int)i, color, 1 );
        circle(drawing, mc[i], 0.3, color, -1);
    }

    /// Show in a window
    imshow("Contours", drawing);

    /// Calculate the area with the moments 00 and compare with the result of the OpenCV function
    cout << "\t Info: Area and Contour Length \n";
    for (size_t i = 0; i < contours.size(); i++)
    {
        cout << " * Contour[" << i << "] - Area (M_00) = " << std::fixed << std::setprecision(2) << mu[i].m00
             << " - Area OpenCV: " << contourArea(contours[i]) << " - Length: " << arcLength(contours[i], true) << endl;
    }
}
