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

using namespace cv;

int main() { 

  // Read image from file
  Mat image = imread("mandrill1.jpg", 1);
  Mat image_copy = imread("mandrill1.jpg", 1);

  // Threshold by looping through all pixels
  for(int y=0; y<image_copy.rows; y++) {
    for(int x=0; x<image_copy.cols; x++) {
      uchar pixelRed = image_copy.at<Vec3b>(y,x)[2];
      int new_x = (x+32) % image_copy.cols;
      int new_y = (y+32) % image_copy.rows;
      image.at<Vec3b>(new_y,new_x)[2] = pixelRed;
    }
  }
  //Save thresholded image
  imwrite("mandrill1_fix.jpg", image);

  return 0;
}
