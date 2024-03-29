/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - load.cpp
// TOPIC: load and display an image
//
// Getting-Started-File for OpenCV
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;           //make available OpenCV namespace

int main() {

  //declare a matrix container to hold an image
  Mat image;

  //load image from a file into the container
  image = imread("mandrillRGB.jpg", CV_LOAD_IMAGE_UNCHANGED);

  for (int y = 0; y < image.rows; y++) //go through all rows (or scanlines)
    for (int x = 0; x < image.cols; x++) { //go through all columns
      uchar pix = image.at<Vec3b>(y,x)[2]; // get the red component
      if(pix > 200) {
        image.at<Vec3b>(y,x)[0] = 255;
        image.at<Vec3b>(y,x)[1] = 255;
        image.at<Vec3b>(y,x)[2] = 255;
      } else {
        image.at<Vec3b>(y,x)[0] = 0;
        image.at<Vec3b>(y,x)[1] = 0;
        image.at<Vec3b>(y,x)[2] = 0;
      }
    }
  //construct a window for image display
  namedWindow("Display window", CV_WINDOW_AUTOSIZE);
  
  //visualise the loaded image in the window
  imshow("Display window", image);

  imwrite("mandrill_rgb.jpg", image);

  //wait for a key press until returning from the program
  waitKey(0);

  //free memory occupied by image 
  image.release();

  return 0;
}
