/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - thr.cpp
// TOPIC: RGB explicit thresholding
//
// Getting-Started-File for OpenCV
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
using namespace std;

using namespace cv;

int main() { 

  // Read image from file
  Mat image = imread("car1.png", CV_LOAD_IMAGE_UNCHANGED);

  cout << image.cols << " " << image.rows << endl;

  Mat image_new_smooth = Mat::zeros(Size(image.cols-1, image.rows-1), CV_8UC1);
  Mat image_new_sharp = Mat::zeros(Size(image_new_smooth.cols-2, image_new_smooth.rows-2), CV_8UC1);

  cout << image_new_smooth.cols << " " << image_new_smooth.rows << endl;

  Mat kernel_lowpass = Mat::ones(3, 3, CV_32F);
  Mat kernel_highpass(3, 3, CV_32F, -1);
  kernel_highpass.at<float>(0,0) = 0;
  kernel_highpass.at<float>(2,2) = 0;
  kernel_highpass.at<float>(2,0) = 0;
  kernel_highpass.at<float>(0,2) = 0;
  kernel_highpass.at<float>(1,1) = 5;
  
  kernel_lowpass = kernel_lowpass/(cv::sum( kernel_lowpass )[0]);
  kernel_highpass = kernel_highpass/(cv::sum( kernel_highpass )[0]);

  // Threshold by looping through all pixels
  for(int y=1; y<image.rows-1; y++) {
   for(int x=1; x<image.cols-1; x++) {
     double pixel = 0;
     for(int i = -1; i<2; i++) {
       for(int j= -1; j<2; j++) {
         pixel += (double) (image.at<uchar>(y-i,x-j) * kernel_highpass.at<float>(i+1,j+1));
       }
     }
     cout << pixel << endl;
     image_new_smooth.at<uchar>(y-1,x-1) = (uchar) (pixel);
} }

//   for(int y=1; y<image_new_smooth.rows-1; y++) {
//    for(int x=1; x<image_new_smooth.cols-1; x++) {
//      int pixel = 0;
//      for(int i = -1; i<2; i++) {
//        for(int j= -1; j<2; j++) {
//          pixel += image_new_smooth.at<uchar>(y-i,x-j) * kernel_highpass.at<float>(i+1,j+1);
//        }
//      }
//      image_new_sharp.at<uchar>(y-1,x-1) = pixel;
// } }

// // Threshold by looping through all pixels
//     for(int y=1; y<image.rows-1; y++) {​​​​​
//         for(int x=1; x<image.cols-1; x++) {​​​​​
//             uchar pixels = image.at<uchar>(y,x) + image.at<uchar>(y-1,x-1) + image.at<uchar>(y-1,x) + image.at<uchar>(y-1,x+1) + image.at<uchar>(y,x-1) 
//             + image.at<uchar>(y,x+1) + image.at<uchar>(y+1,x-1) + image.at<uchar>(y+1,x) + image.at<uchar>(y+1,x+1);
//             image2.at<uchar>(y,x) = pixels/9;
//         }​​​​​
//     }​​​​​
  cout << "kernel: "<< endl<< kernel_highpass << endl<< endl;
  //Save thresholded image
  imwrite("mandrill_convolve.jpg", image_new_smooth);

  return 0;
}
