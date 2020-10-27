// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup


using namespace cv;

void GaussianBlur(
	cv::Mat &input, 
	int size,
	cv::Mat &blurredOutput);

void addImage(cv::Mat &input, cv::Mat &addedInput);

void subImage(cv::Mat &input, cv::Mat &sub, cv::Mat &subInput);

int main( int argc, char** argv )
{

 // LOADING THE IMAGE
 char* imageName = argv[1];

 Mat image;
 image = imread( imageName, 1 );

 if( argc != 2 || !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }

 // CONVERT COLOUR, BLUR AND SAVE
 Mat gray_image;
 cvtColor( image, gray_image, CV_BGR2GRAY );

 Mat carBlurred;
 GaussianBlur(gray_image,15,carBlurred);

 Mat carCopy;
 addImage(gray_image, carCopy);

 Mat carSub;
 subImage(carCopy, carBlurred, carSub);

 imwrite( "sharp.jpg", carSub );

 return 0;
}

void addImage(cv::Mat &input, cv::Mat &addedInput) {
	addedInput.create(input.size(), input.type());

	for(int i = 0; i<input.rows; i++) {
		for(int j=0; j<input.cols; j++) {
			int val = (int) input.at<uchar>(i,j);
			addedInput.at<uchar>(i,j) = (uchar) val*2;
		}
	}
}

void subImage(cv::Mat &input, cv::Mat &sub, cv::Mat &subInput) {
	subInput.create(input.size(), input.type());

	for(int i = 0; i<input.rows; i++) {
		for(int j=0; j<input.cols; j++) {
			int val = (int) input.at<uchar>(i,j);
			int val2 = (int) sub.at<uchar>(i,j);
			subInput.at<uchar>(i,j) = (uchar) val - val2;
		}
	}
}

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D 
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);
	
	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{	
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;							
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar) sum;
		}
	}
}
