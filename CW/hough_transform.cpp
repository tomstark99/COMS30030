// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

#define pi 3.14159265358979323846

using namespace cv;
using namespace std;

void sobel(Mat &input, Mat &output_x, Mat &output_y, Mat &output_mag, Mat &output_dir);
void normalise(Mat &input, string num);
void threshold(Mat &input, int t, Mat &output);
void gaussian(Mat &input, int size, Mat &output);
void filter_non_max(Mat &input_mag, Mat &input_dir);
vector<vector<int> > h_transform(Mat &input, int r_min, int r_max, double threshold, Mat &direction);
void draw_circles(Mat &input, vector<vector<int> > circles);

int ***malloc3dArray(int dim1, int dim2, int dim3) {
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));
 
    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
	    for (j = 0; j < dim2; j++) {
  	        array[i][j] = (int *) malloc(dim3 * sizeof(int));
	    }
 
    }
    return array;
}

int main( int argc, char** argv ) {

	char* imageName = argv[1];

	Mat image;
	image = imread( imageName, 1 );

	if( argc != 2 || !image.data ) {
		printf( " No image data \n " );
		return -1;
	}

 	Mat img_gray;
 	cvtColor( image, img_gray, CV_BGR2GRAY );

	Mat img_blur;
	// set number for gaussian kernel size, different kernel size works better with different pic
	gaussian(img_gray, 7, img_blur);

	Mat img_x(image.size(), CV_32FC1);
	Mat img_y(image.size(), CV_32FC1);
	Mat img_magnitude(image.size(), CV_32FC1);
	Mat img_direction(image.size(), CV_32FC1);
	// output normalised magnitude and gradient image
	sobel(img_blur, img_x, img_y, img_magnitude, img_direction); 

	Mat r_img_x(image.size(), CV_8UC1);
	Mat r_img_y(image.size(), CV_8UC1);
	Mat r_img_magnitude(image.size(), CV_8UC1, Scalar(0));
	Mat r_img_direction(image.size(), CV_8UC1, Scalar(0));

	normalize(img_x,r_img_x,0,255,NORM_MINMAX, CV_8UC1);
    normalize(img_y,r_img_y,0,255,NORM_MINMAX, CV_8UC1);
    normalize(img_magnitude,r_img_magnitude,0,255,NORM_MINMAX);
    normalize(img_direction,r_img_direction,0,255,NORM_MINMAX);
    // imwrite("dart_x.jpg",r_img_x);
    // imwrite("dart_y.jpg",r_img_y);
    imwrite("dart_mag.jpg",r_img_magnitude);
    // imwrite("dart_dir.jpg", r_img_direction);

	// filter out some non-max values, this seems to improve detection ?
	// filter_non_max(img_magnitude, img_direction);

	Mat img_threshold = imread("dart_mag.jpg", 1);
    Mat gray_test;
    cvtColor( img_threshold, gray_test, CV_BGR2GRAY );

	// set threshold (between 0 and 255) for the normalised magnitude image
	threshold(gray_test, 15, img_threshold);

	vector<vector<int> > circles = h_transform(img_threshold, 40, min(img_threshold.rows,img_threshold.cols), 15, img_direction);
	draw_circles(image, circles);

 	return 0;
}

void sobel(Mat &input, Mat &output_x, Mat &output_y, Mat &output_mag, Mat &output_dir) {

	Mat kX = Mat::ones(3, 3, CV_32F);
	
	// creating the sobel kernel for x
	kX.at<float>(0,0) = -1;
	kX.at<float>(1,0) = -2;
	kX.at<float>(0,1) = 0;
	kX.at<float>(1,1) = 0;
	kX.at<float>(1,2) = 2;
	kX.at<float>(1,2) = 2;
	kX.at<float>(2,0) = -1;
	kX.at<float>(2,1) = 0;

	// sobel kernel for y
	Mat kY = kX.t();

	int kernelRadiusX = ( kX.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kX.size[1] - 1 ) / 2;

	Mat paddedInput;
	copyMakeBorder( input, paddedInput, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, BORDER_REPLICATE );

	for ( int i = 0; i < input.rows; i++ ) {	
		for( int j = 0; j < input.cols; j++ ) {
			float sum_x = 0.0;
			float sum_y = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					float imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					float kernel_x = kX.at<float>( kernelx, kernely );
					float kernel_y = kY.at<float>( kernelx, kernely );

					sum_x += imageval * kernel_x;
					sum_y += imageval * kernel_y;
				}
			}
			output_x.at<float>(i, j) = (float) sum_x;
			output_y.at<float>(i, j) = (float) sum_y;
			output_mag.at<float>(i, j) = (float) sqrt((sum_y*sum_y) + (sum_x*sum_x));
			output_dir.at<float>(i, j) = (float) atan2(sum_y, sum_x);
		}
	}
}

void normalise(Mat &input, string num) {
	double min; 
	double max; 

	Mat output;
	output.create(input.size(), CV_8UC1);

	minMaxLoc( input, &min, &max );

	for(int i = 0; i < input.rows; i++) {
		for(int j = 0; j < input.cols; j++) {
			float val = (float) input.at<float>(i, j);
			output.at<uchar>(i,j) = (uchar) (val - min)*((255)/max-min);
		}
	}
	// imwrite( "dart_norm_" + num + ".jpg", input );
}

void gaussian(Mat &input, int size, Mat &output)
{
	output.create(input.size(), input.type());

	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);

	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	// cout << "kernel: "<< endl<< kernel << endl<< endl;

	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	for ( int i = 0; i < input.rows; i++ ) {	
		for( int j = 0; j < input.cols; j++ ) {
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
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
			output.at<uchar>(i, j) = (uchar) sum;
		}
	}
}

void filter_non_max(Mat &input_mag, Mat &input_dir) {
	assert(input_mag.size() == input_dir.size() && input_mag.type() == input_dir.type());

	for(int i = 1; i < input_mag.rows-1; i++) {
		for(int j = 1; j < input_mag.cols-1; j++) {
			double angle;
			if(input_dir.at<uchar>(i,j) >= 0) {
				angle = double(input_dir.at<uchar>(i,j));
			} else{
				angle = double(input_dir.at<uchar>(i,j)) + pi;
			}
			int r_angle = round(angle / (pi / 4));
			int mag = input_mag.at<uchar>(i,j);
			if((r_angle == 0 || r_angle == 4) && (input_mag.at<uchar>(i-1,j) > mag || input_mag.at<uchar>(i+1,j) > mag) || (r_angle == 1 && (input_mag.at<uchar>(i-1,j-1) > mag || input_mag.at<uchar>(i+1,j+1) > mag)) || (r_angle == 2 && (input_mag.at<uchar>(i,j-1) > mag || input_mag.at<uchar>(i,j+1) > mag)) || (r_angle == 3 && (input_mag.at<uchar>(i+1,j-1) > mag || input_mag.at<uchar>(i-1,j+1) > mag))) {
				input_mag.at<uchar>(i,j) = (uchar) 0;
			}
		}
	}
}

void threshold(Mat &input, int t, Mat &output) {
	assert(t >= 0 && t <= 255);
	output.create(input.size(), input.type());
	for(int i = 0; i < input.rows; i++) {
		for(int j = 0; j < input.cols; j++) {
			int val = (int) input.at<uchar>(i, j);
			if(val > t) {
				output.at<uchar>(i,j) = (uchar) 255;
			} else {
				output.at<uchar>(i,j) = (uchar) 0;
			}
		}
	}
	imwrite("dart_threshold.jpg", output);
}

vector<vector<int> > h_transform(Mat &input, int r_min, int r_max, double threshold, Mat &direction) {

	int ***hough_space = malloc3dArray(input.rows, input.cols, r_max);
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            for (int r = 0; r < r_max; r++) {
                hough_space[i][j][r] = 0;
            }
        }
    }
    for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
			if(input.at<uchar>(x,y) == 255) {
				for (int r = 0; r < r_max; r++) {
					int xc = int(r * sin(direction.at<float>(x,y)));
					int yc = int(r * cos(direction.at<float>(x,y)));

					int a = x - xc;
					int b = y - yc;
					int c = x + xc;
					int d = y + yc;
					if(a >= 0 && a < input.rows && b >= 0 && b < input.cols) {
						hough_space[a][b][r] += 1;
					}
					if(c >= 0 && c < input.rows && d >= 0 && d < input.cols) {
						hough_space[c][d][r] += 1;
					}
				}
			}
        }
    }

	Mat hough_output(input.rows, input.cols, CV_32FC1);
 
    for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
            for (int r = r_min; r < r_max; r++) {
                hough_output.at<float>(x,y) += hough_space[x][y][r];
            }
 
        }
    }

	Mat hough_norm(input.rows, input.cols, CV_8UC1);
    normalize(hough_output, hough_norm, 0, 255, NORM_MINMAX);
 
    imwrite( "hough.jpg", hough_norm );

	vector<vector<int> > circles;
	for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
			bool test_pass = true;
			map<int, int> t_circles;
            for (int r = r_min; r < r_max; r++) {
				if(hough_space[x][y][r] > threshold) {
					t_circles[r] = hough_space[x][y][r];
				}
            }
			int max_c = 0;
			int max_r = 0;
			// for(int i = 0; i < circles.size(); i++) {
			// 	vector<int> circle = circles[i];
			// 	int xc = circle[0];
			// 	int yc = circle[1];
			// 	int rc = circle[2];

			// 	if(!(pow((x-xc),2) + pow((y-yc),2) > pow(rc,2))) {
			// 		test_pass = false;
			// 	}
			// }
			for(map<int, int>::const_iterator it = t_circles.begin(); it != t_circles.end(); ++it) {
				// if(it->second > max_c) {
				// 	max_r = it->first;
				// 	max_c = it->second;
				// }
				for(int i = 0; i < circles.size(); i++) {
					vector<int> circle = circles[i];
					int r = circle[2];
					if(r - 5 < it->first && r+5 > it->first){
						test_pass = false;
					}
				}
				if(test_pass) {
					vector<int> circle;
					circle.push_back(x);
					circle.push_back(y);
					circle.push_back(it->first);
					// cout << "radius: " << it->first << endl;
					circles.push_back(circle);
				}
			}
			// if(hough_space[x][y][max_r] > threshold && test_pass) {
			// 	vector<int> circle;
			// 	circle.push_back(x);
			// 	circle.push_back(y);
			// 	circle.push_back(max_r);
			// 	circles.push_back(circle);
			// }
        }
    }

	cout << "circles: " << circles.size() << endl;

	return circles;
}

void draw_circles(Mat &input, vector<vector<int> > circles) {

	for(int i = 0; i < circles.size(); i++) {
		vector<int> c = circles[i];
		Point center = Point(c[1], c[0]);
		circle(input, center, 1, Scalar(0, 255, 0), 3, 8, 0);
		int radius = c[2];
		circle(input, center, radius, Scalar(0, 0, 255), 2, 8, 0);
	}

	stringstream ss;
	ss << (int) circles.size();
	imwrite("detected_circles_"+ss.str()+".jpg", input);

}
