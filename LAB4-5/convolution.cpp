// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup


using namespace cv;
using namespace std;

void sobel(Mat &input);
void normalise(Mat &input, string num);

int main( int argc, char** argv ) {

	// LOADING THE IMAGE
	char* imageName = argv[1];

	Mat image;
	image = imread( imageName, 1 );

	if( argc != 2 || !image.data ) {
		printf( " No image data \n " );
		return -1;
	}

 	// CONVERT COLOUR, BLUR AND SAVE
 	Mat gray_image;
 	cvtColor( image, gray_image, CV_BGR2GRAY );

	sobel(gray_image);

	

 	return 0;
}

void sobel(Mat &input)
{
	// intialise the output using the input
	Mat output_x;
	Mat output_y;
	Mat output_mag;
	Mat output_dir;
	output_x.create(input.size(), input.type());
	output_y.create(input.size(), input.type());
	output_mag.create(input.size(), input.type());
	output_dir.create(input.size(), input.type());

	Mat kX = Mat::ones(3, 3, CV_32F);
	kX.at<float>(0,0) = -1;
	kX.at<float>(1,0) = -2;
	kX.at<float>(0,1) = 0;
	kX.at<float>(1,1) = 0;
	kX.at<float>(1,2) = 2;
	kX.at<float>(1,2) = 2;
	kX.at<float>(2,0) = -1;
	kX.at<float>(2,1) = 0;

	Mat kY = kX.t();

	int kernelRadiusX = ( kX.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kX.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ ) {	
		for( int j = 0; j < input.cols; j++ ) {
			double sum_x = 0.0;
			double sum_y = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					double imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernel_x = kX.at<double>( kernelx, kernely );
					double kernel_y = kY.at<double>( kernelx, kernely );

					// do the multiplication
					sum_x += imageval * kernel_x;
					sum_y += imageval * kernel_y;
				}
			}
			// set the output value as the sum of the convolution
			output_x.at<uchar>(i, j) = (uchar) sum_x;
			output_y.at<uchar>(i, j) = (uchar) sum_y;
			output_mag.at<uchar>(i, j) = (uchar) sqrt((sum_y*sum_y) + (sum_x*sum_x));
			output_dir.at<uchar>(i, j) = (uchar) atan2(sum_y, sum_x);
		}
	}

	imwrite( "coin_x.jpg", output_x );
	imwrite( "coin_y.jpg", output_y );
	imwrite( "coin_mag.jpg", output_mag );
	imwrite( "coin_dir.jpg", output_dir );

	normalise(output_x, "x");
	normalise(output_y, "y");
	normalise(output_mag, "mag");
	normalise(output_dir, "dir");
}

void normalise(Mat &input, string num) {
	double min; 
	double max; 
	minMaxLoc( input, &min, &max );

	for(int i = 0; i < input.rows; i++) {
		for(int j = 0; j < input.cols; j++) {
			double val = (double) input.at<uchar>(i, j);
			input.at<uchar>(i,j) = (uchar) (val - min)*((255)/max-min);
		}
	}
	imwrite( "coin_norm_" + num + ".jpg", input );
}
