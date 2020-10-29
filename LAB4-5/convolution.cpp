// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#define pi 3.14159265358979323846

using namespace cv;
using namespace std;

void sobel(Mat &input, Mat &output_mag, Mat &output_dir);
void normalise(Mat &input, string num);
void threshold(Mat &input, int t, Mat &output);
void gaussian(Mat &input, int size, Mat &output);
void filter_non_max(Mat &input_mag, Mat &input_dir);
int hough_transform(Mat &input, int r_min, int r_max, double threshold, Mat &output);
int hough_transform_2(Mat &input, int r_min, int r_max, double threshold, Mat &output);
vector<Point3d> HoughTransform(cv::Mat &input, cv::Mat &gradient_dir);

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

	Mat image_blr;
	gaussian(gray_image, 7, image_blr);

	Mat image_mag;
	Mat image_dir;
	sobel(image_blr, image_mag, image_dir);

	filter_non_max(image_mag, image_dir);

	Mat image_thr;
	threshold(image_mag, 50, image_thr);
	imwrite("coin_threshold.jpg", image_thr);

	Mat canny = imread("canny.png", CV_LOAD_IMAGE_UNCHANGED);
	cvtColor( canny, canny, CV_BGR2GRAY );

	Mat image_hou;
	// int c = hough_transform_2(image_thr, 35, 45, 0.5, image_hou);
	int c = hough_transform_2(image_thr, 40, 90, 0.7, image_hou);
	stringstream ss;
	ss << c;
	imwrite("detected_circles_"+ss.str()+".jpg", image_hou);

	// vector<Point3d> alie = HoughTransform(image_thr,image_dir);
	// imwrite("stefnib.jpg", image_thr);

 	return 0;
}

void sobel(Mat &input, Mat &output_mag, Mat &output_dir) {
	// intialise the output using the input
	Mat output_x;
	Mat output_y;
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
	// imwrite( "coin_norm_" + num + ".jpg", input );
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
}

int hough_transform(Mat &input, int r_min, int r_max, double threshold, Mat &output) {
	output.create(input.size(), input.type());

	vector<vector<int> > points;

	int steps = 100;

	for(int i = r_min; i < r_max+1; i++) {
		// vector<vector<int> > temp_1;
		for(int j = 0; j < steps; j++) {
			vector<int> temp;
			temp.push_back(i);
			temp.push_back(int(i * cos(2 * pi * j / steps)));
			temp.push_back(int(i * sin(2 * pi * j / steps)));
			points.push_back(temp);
		}
		// points.push_back(temp_1);
	}

	// for(int i = 0; i < points.size(); i++) {
	// 	for(int j = 0; j < points[i].size(); j++) {
	// 		cout << points[i][j] << ' ';
	// 	}
	// 	cout << endl;
	// }

	// cout << points.size() << endl;
	// assert(points.size() == 3);

	// cout << "img width: " << input.rows << " img height: " << input.cols << endl;

	map<vector<int>, int> acc;

	// vector<pair<int, int> > pixels;

	// for(int x = 0; x < input.rows; x++) {
	// 	for( int y = 0; y < input.cols; y++) {
	// 		int val = (int) input.at<uchar>(x,y);
	// 		if(val > 75) {
	// 			pixels.push_back(make_pair(x,y));
	// 		}
	// 	}
	// }

	// cout << "pixels: " << pixels.size() << endl;

	// for(int i = 0; i < pixels.size(); i++) {
	// 	pair<int, int> p = pixels[i];
	// 	cout << p.first << ' ' << p.second << endl;
	// }
	int percent_latest = 0;
	for(int i = 0; i < input.rows; i++) {
		double percent = (double) i/input.rows;
		percent *= 100;
		if((int) percent != percent_latest) {
			percent_latest = (int) percent;
			cout << "creating map: " << (int) percent << '%' << endl;
		}

		for(int j = 0; j < input.cols; j++) {
			if(input.at<uchar>(i,j) > 75) {
				// cout << i << ' ' << j << endl;
				for(int r = 0; r < points.size(); r++) {		
					vector<int> point = points[r];
					int a = i - point[1];
					int b = j - point[2];
					vector<int> temp;
					temp.push_back(a);
					temp.push_back(b);
					temp.push_back(point[0]);
					acc[temp] += 1;
					// if( i < input.rows && j < 10) {
					// 	cout << a << ' ' << b << ' ' << points[r][0] << endl; 
					// }
				}
			}
		}
	}

	cout << "map size: " << acc.size() << endl;

	// for(map<vector<int>, int>::const_iterator it = acc.begin(); it != acc.end(); ++it) {
	// 	for(int i = 0; i < it->first.size(); i++) {
	// 		cout << it->first[i] << ' ';
	// 	}
	// 	cout << ": " << it->second << endl;
	// }

	vector<vector<int> > circles;
	for(map<vector<int>, int>::const_iterator it = acc.begin(); it != acc.end(); ++it) {
		bool all_check = true;
		int x = it->first[0];
		int y = it->first[1];
		int r = it->first[2];
		for(int j = 0; j < circles.size(); j++) {
			int x_circ = (x - circles[j][0]);
			int y_circ = (y - circles[j][1]);
			int r_circ = (circles[j][2]);
			if(!((x_circ*x_circ)+(y_circ*y_circ) > (r_circ*r_circ))) {
				all_check = false;
			}
		}
		double t_thresh = (double) it->second/steps;
		if(t_thresh >= threshold && all_check) {
			cout << t_thresh << ' ' << x << ' ' << y << ' ' << r << endl;
			vector<int> temp;
			temp.push_back(x);
			temp.push_back(y);
			temp.push_back(r);
			circles.push_back(temp);
		}
	}
	cout << "circles: " << circles.size() << endl;
 	cvtColor( input, input, CV_GRAY2BGR );
	for(int i = 0; i < circles.size(); i++) {
		vector<int> c = circles[i];
		Point center = Point(c[0], c[1]);
		circle(input, center, 1, Scalar(0, 255, 0), 1, 8, 0);
		int radius = c[2];
		circle(input, center, radius, Scalar(255, 0, 255), 1, 8, 0);
	}
	// imshow("detected circles", input);
	// waitKey();
	output = input.clone();
	return (int) circles.size();
	return 0;
}

int hough_transform_2(Mat &input, int r_min, int r_max, double threshold, Mat &output) {
	output.create(input.size(), input.type());
	int steps = 10;

	vector<vector<int> > points;
	for(int i = r_min; i < r_max+1; i++) {
		for(int j = 0; j < steps; j++) {
			vector<int> temp;
			temp.push_back(i);
			temp.push_back(int(i * cos(2 * pi * j / steps)));
			temp.push_back(int(i * sin(2 * pi * j / steps)));
			points.push_back(temp);
		}
	}

	vector<pair<int, int> > pixels;
	for(int x = 0; x < input.rows; x++) {
		for( int y = 0; y < input.cols; y++) {
			int val = (int) input.at<uchar>(x,y);
			if(val > 75) {
				pixels.push_back(make_pair(x,y));
			}
		}
	}

	int percent_latest = 0;
	map<vector<int>, int> acc;
	for(int i = 0; i < pixels.size(); i++) {
		double percent = (double) i/pixels.size();
		percent *= 100;
		if((int) percent != percent_latest) {
			percent_latest = (int) percent;
			cout << "creating map: " << (int) percent << '%' << endl;
		}
		pair<int, int> pix = pixels[i];
		int x = pix.first;
		int y = pix.second;
		for(int j = 0; j < points.size(); j++) {
			vector<int> temp;
			vector<int> p = points[j];
			int a = x - p[1];
			int b = y - p[2];
			temp.push_back(a);
			temp.push_back(b);
			temp.push_back(p[0]);
			acc[temp] += 1;
		}
	}
	cout << "map size: " << acc.size() << endl;

	vector<vector<int> > circles;
	for(map<vector<int>, int>::const_iterator it = acc.begin(); it != acc.end(); ++it) {
		bool test_pass = true;
		vector<int> key = it->first;

		int x = key[0];
		int y = key[1];
		int r = key[2];
		for(int i = 0; i < circles.size(); i++) {
			vector<int> circle = circles[i];
			int xc = circle[0];
			int yc = circle[1];
			int rc = circle[2];

			if(!(pow((x-xc),2) + pow((y-yc),2) > pow(rc,2))) {
				test_pass = false;
			}
		}
		double t_thresh = (double) it->second/steps;
		if(t_thresh >= threshold && test_pass) {
			cout << t_thresh << ' ' << x << ' ' << y << ' ' << r << endl;
			vector<int> temp;
			temp.push_back(x);
			temp.push_back(y);
			temp.push_back(r);
			circles.push_back(temp);
		}
	}
	cout << "circles: " << circles.size() << endl;

 	cvtColor( input, input, CV_GRAY2BGR );
	// input = input.t();
	for(int i = 0; i < circles.size(); i++) {
		vector<int> c = circles[i];
		Point center = Point(c[1], c[0]);
		circle(input, center, 1, Scalar(0, 255, 0), 3, 8, 0);
		int radius = c[2];
		circle(input, center, radius, Scalar(0, 0, 255), 2, 8, 0);
	}

	output = input.clone();
	return (int) circles.size();
}

vector<Point3d> HoughTransform(cv::Mat &input, cv::Mat &gradient_dir)
{
    int rLen = input.rows / 2;
    int dims[] = {input.rows, input.cols, rLen};
    Mat houghSpace(3, dims, CV_8UC(1), Scalar::all(0));

    for (int y = 0; y < input.rows; y++)
    {
        for (int x = 0; x < input.cols; x++)
        {
            for (int r = 20; r < rLen; r++)
            {
                int a = r * std::cos(gradient_dir.at<double>(y, x));
                int b = r * std::sin(gradient_dir.at<double>(y, x));
                int x0 = x - a;
                int y0 = y - b;
                if (x0 >= 0 && x0 < input.cols && y0 >= 0 && y0 < input.rows)
                    houghSpace.at<cv::Vec3i>(y0, x0)[r]++;

                x0 = x + a;
                y0 = y + b;
                if (x0 >= 0 && x0 < input.cols && y0 >= 0 && y0 < input.rows)
                    houghSpace.at<cv::Vec3i>(y0, x0)[r]++;
            }
        }
    }
 	cvtColor( input, input, CV_GRAY2BGR );
    vector <Point3d> circles;
    for (int y = 0; y < input.rows; y++)
    {
        for (int x = 0; x < input.cols; x++)
        {
            for (int r = 0; r < rLen; r++)
            {
                if (houghSpace.at<cv::Vec3i> (y, x)[r] > 100)
                {
                    if(circles.size()<30) {
                    Point3d temp(x, y, r);
                    circles.push_back(temp);
                    circle(input, Point(x, y), r, Scalar(0, 0, 255), 1, 8, 0);
                    }
                }
            }
        }
    }
	cout << circles.size() << endl;
    return circles;
}
