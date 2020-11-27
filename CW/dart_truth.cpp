/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>

#define pi 3.14159265358979323846

using namespace std;
using namespace cv;

/** Function Headers */
vector<string> split( const string &line, char delimiter );
vector<Rect> read_csv(string num);
float get_iou(Rect t, Rect d);
void detectAndDisplay( Mat frame, vector<Rect> truths, string num, vector<vector<int>> circles );
float get_f1_score(float t_p, float f_p, float f_n);
void sobel(Mat &input, Mat &output_x, Mat &output_y, Mat &output_mag, Mat &output_dir);
void threshold(Mat &input, int t, Mat &output, string num, string ver);
void gaussian(Mat &input, int size, Mat &output);
void filter_non_max(Mat &input_mag, Mat &input_dir);
void lh_transform(Mat &input, Mat &direction, string num);
vector<vector<int>> ch_transform(Mat &input, int r_min, int r_max, double threshold, Mat &direction, string num);
void draw_circles(Mat &input, vector<vector<int> > circles, string num);

/** Global variables */
String cascade_name = "dart_cascade/cascade.xml";
CascadeClassifier cascade;

int **malloc2dArray(int dim1, int dim2) {
    int i, j;
    int **array = (int **) malloc(dim1 * sizeof(int *));
 
    for (i = 0; i < dim1; i++) {
        array[i] = (int *) malloc(dim2 * sizeof(int));
    }
    return array;
}
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

/** @function main */
int main( int argc, const char** argv )
{
	string image_n = argv[1];

	Mat frame = imread("source_images/dart"+image_n+".jpg", CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	Mat img_gray, img_blur;
 	cvtColor( frame, img_gray, CV_BGR2GRAY );
	gaussian(img_gray, 7, img_blur);

	Mat img_x(frame.size(), CV_32FC1);
	Mat img_y(frame.size(), CV_32FC1);
	Mat img_magnitude(frame.size(), CV_32FC1);
	Mat img_direction(frame.size(), CV_32FC1);
	Mat r_img_x(frame.size(), CV_8UC1);
	Mat r_img_y(frame.size(), CV_8UC1);
	Mat r_img_magnitude(frame.size(), CV_8UC1, Scalar(0));
	Mat r_img_direction(frame.size(), CV_8UC1, Scalar(0));

	sobel(img_blur, img_x, img_y, img_magnitude, img_direction); 
	normalize(img_x,r_img_x,0,255,NORM_MINMAX, CV_8UC1);
    normalize(img_y,r_img_y,0,255,NORM_MINMAX, CV_8UC1);
    normalize(img_magnitude,r_img_magnitude,0,255,NORM_MINMAX);
    normalize(img_direction,r_img_direction,0,255,NORM_MINMAX);
    imwrite("detected_darts/"+image_n+"/x.jpg",r_img_x);
    imwrite("detected_darts/"+image_n+"/y.jpg",r_img_y);
    imwrite("detected_darts/"+image_n+"/magnitude.jpg",r_img_magnitude);
    imwrite("detected_darts/"+image_n+"/direction.jpg", r_img_direction);

	Mat img_threshold = imread("detected_darts/"+image_n+"/magnitude.jpg", 1);
    Mat gray_test;
    cvtColor( img_threshold, gray_test, CV_BGR2GRAY );

	// set threshold (between 0 and 255) for the normalised magnitude image
	threshold(gray_test, 30, img_threshold, image_n, "source");

	vector<vector<int>> circles = ch_transform(img_threshold, 40, min(img_threshold.rows,img_threshold.cols), 15, img_direction, image_n);
	lh_transform(img_threshold, img_direction, image_n);
	// 3. Detect Faces and Display Result
	draw_circles(frame, circles, image_n);

	detectAndDisplay( frame, read_csv(image_n), image_n, circles );

	// 4. Save Result Image
	imwrite( "detected_darts/"+image_n+"/detected_filtered.jpg", frame );

	return 0;
}

std::vector<std::string> split(const std::string &line, char delimiter) {
	auto haystack = line;
	std::vector<std::string> tokens;
	size_t pos;
	while ((pos = haystack.find(delimiter)) != std::string::npos) {
		tokens.push_back(haystack.substr(0, pos));
		haystack.erase(0, pos + 1);
	}
	// Push the remaining chars onto the vector
	tokens.push_back(haystack);
	return tokens;
}

vector<Rect> read_csv(string num) {

	vector<Rect> truths;

	string file_name = "dart_truths/"+num+".csv";
	ifstream file(file_name);
	string line;

	while(getline(file, line)) {
		vector<string> tokens = split(line, ',');
		truths.push_back(Rect(stoi(tokens[0]),stoi(tokens[1]),stoi(tokens[2]),stoi(tokens[3])));
	}
	file.close();

	return truths;
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

void gaussian(Mat &input, int size, Mat &output) {
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

void threshold(Mat &input, int t, Mat &output, string num, string ver) {
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
	imwrite("detected_darts/"+num+"/threshold_"+ver+".jpg", output);
}

void lh_transform(Mat &input, Mat &direction, string num) {

	int diag = sqrt(pow(input.rows,2)+pow(input.cols,2));

	int **hough_space = malloc2dArray(diag,360);
    for (int i = 0; i < diag; i++) {
        for (int j = 0; j < 360; j++) {
            hough_space[i][j] = 0;
        }
    }
    for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
			if(input.at<uchar>(x,y) == 255) {
				//for (int r = 0; r < r_max; r++) {
					int th = int(direction.at<float>(x,y)*(180/pi)) + 180;
					for(int t = th-5; t <= th+5; t++) {
						int mod_th = (t+360) % 360;
						float t_rad = (mod_th-180)*(pi/180);
						int xc = int(x * sin(t_rad));
						int yc = int(y * cos(t_rad));
						int p = xc + yc;
						if(p >= 0 && p <= diag) {
							hough_space[p][mod_th] += 1;
						}
					}
			}
        }
    }

	Mat hough_output(diag, 360, CV_32FC1, Scalar(0));
 
    for (int p = 0; p < diag; p++) {
        for (int t = 0; t < 360; t++) {
			hough_output.at<float>(p,t) = hough_space[p][t];
        }
    }

	Mat hough_norm(diag, 360, CV_8UC1, Scalar(0));
    normalize(hough_output, hough_norm, 0, 255, NORM_MINMAX);
	
    imwrite("detected_darts/"+num+"/rho_theta_space.jpg", hough_norm );

	Mat img_threshold = imread("detected_darts/"+num+"/rho_theta_space.jpg", 1);
    Mat gray_test;
    cvtColor( img_threshold, gray_test, CV_BGR2GRAY );

	// set threshold (between 0 and 255) for the normalised magnitude image
	threshold(gray_test, 10, img_threshold, num, "rho_theta");

	Mat hough_output_o(input.rows, input.cols, CV_32FC1, Scalar(0));
 
	for(int p = 0; p < hough_output.rows; p++) {
		for(int th = 0; th < hough_output.cols; th++) {
			if(img_threshold.at<uchar>(p,th) == 255) {
				float t_rad = (th-180) * (pi/180);
				for(int x = 0; x < input.cols; x++) {
					int y = ((-cos(t_rad))/sin(t_rad))*x + (p/sin(t_rad));

					if(y >= 0 && y < input.rows) {
						hough_output_o.at<float>(y,x)++;
					}
				}
			}
		}
	}

	Mat hough_norm_o(input.rows, input.cols, CV_8UC1, Scalar(0));
    normalize(hough_output_o, hough_norm_o, 0, 255, NORM_MINMAX);
	Mat hough_th;
 
    imwrite("detected_darts/"+num+"/hough_space_lines.jpg", hough_norm_o );

	Mat img_threshold_o = imread("detected_darts/"+num+"/hough_space_lines.jpg", 1);
    Mat gray_test_o;
    cvtColor( img_threshold_o, gray_test_o, CV_BGR2GRAY );

	// set threshold (between 0 and 255) for the normalised magnitude image
	threshold(gray_test_o, 160, img_threshold_o, num, "lines");

	cout << "finished " << num << endl;
}

vector<vector<int> > ch_transform(Mat &input, int r_min, int r_max, double threshold, Mat &direction, string num) {

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
 
    imwrite("detected_darts/"+num+"/hough_space_circles.jpg", hough_norm );

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

void draw_circles(Mat &input, vector<vector<int> > circles, string num) {
	Mat output = input.clone();
	for(int i = 0; i < circles.size(); i++) {
		vector<int> c = circles[i];
		Point center = Point(c[1], c[0]);
		circle(output, center, 1, Scalar(0, 255, 0), 3, 8, 0);
		int radius = c[2];
		circle(output, center, radius, Scalar(0, 0, 255), 2, 8, 0);
	}
	string xd = to_string(circles.size());
	imwrite("detected_darts/"+num+"/detected_circles_"+xd+".jpg", output);

}

//https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
float get_iou(Rect t, Rect d) {
	float width = min(d.x + d.width, t.x + t.width) - max(d.x, t.x);
	float height = min(d.y + d.height, t.y + t.height) - max(d.y, t.y);

	float int_area = width * height;
	float uni_area = (d.width * d.height) + (t.width * t.height) - int_area;

	return int_area/uni_area;
}

//https://en.wikipedia.org/wiki/F-score
float get_f1_score(float t_p, float f_p, float f_n) {
	return (t_p == 0 && f_p == 0 && f_n == 0) ? 0 : t_p/(t_p + 0.5 * (f_p+f_n));
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, vector<Rect> truths, string num, vector<vector<int>> circles ) {

	std::vector<Rect> darts;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, darts, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	float iou_threshold = 0.5;

	// cout << "darts pre remove: " <<darts.size() << endl;

	Mat lines = imread("detected_darts/"+num+"/threshold_lines.jpg", 1);
	Mat lines_g;
    cvtColor( lines, lines_g, CV_BGR2GRAY );
	assert(lines_g.rows == frame.rows && lines_g.cols == frame.cols);

	Mat frame2 = frame.clone();

	for(int c = 0; c < circles.size(); c++) {
		circle(frame, Point(circles[c][1],circles[c][0]), 1, Scalar(0, 255, 255), 3, 8, 0);
		circle(frame2, Point(circles[c][1],circles[c][0]), 1, Scalar(0, 255, 255), 3, 8, 0);
	}

	vector<Rect> darts_filtered;
	for(int i = 0; i < darts.size(); i++) {
		int in = 0;
		int l_in = 0;
		for(int c = 0; c < circles.size(); c++) {
			int c_x = circles[c][1], c_y = circles[c][0];
			// circle(frame, Point(c_x,c_y), 1, Scalar(0, 255, 255), 3, 8, 0);
			for(int x = darts[i].x; x < darts[i].x+darts[i].width; x++ ) {
				for(int y = darts[i].y; y < darts[i].y+darts[i].height; y++) {
					if(lines_g.at<uchar>(y,x) == 255) { l_in++; }
					// circle(frame, Point(x,y), 1, Scalar(0, 0, 255), 3, 8, 0);
				}
			}
					if(c_x > darts[i].x && c_x < darts[i].x+darts[i].width && c_y > darts[i].y && c_y < darts[i].y+darts[i].height) { in++; }
		}
		if(l_in != 0) cout << "rectangle " << darts[i] << " detected white pixels" << endl;
		if(in >= 2 || l_in >= 20) {
			// for(int t = 0; t < truths.size(); t++) {
				// if(get_iou(truths[t], darts[i]) > iou_threshold){
			// cout << "rectangle allowed" << endl;
			darts_filtered.push_back(darts[i]);
			// rectangle(frame, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar( 0, 255, 0 ), 2);
				// }
			// }
		}
	}

	int avg_x = 0, avg_y = 0, avg_w = 0, avg_h = 0;
	for(int i = 0; i < darts_filtered.size(); i++) {
		cout << darts_filtered[i] << endl;
		avg_x += darts_filtered[i].x;
		avg_y += darts_filtered[i].y;
		avg_w += darts_filtered[i].width;
		avg_h += darts_filtered[i].height;
	}
	cout << avg_w << " x " << avg_h << " at " << avg_x << "," << avg_y << endl;
	Rect dart_avg(int(avg_x/darts_filtered.size()),int(avg_y/darts_filtered.size()),int(avg_w/darts_filtered.size()),int(avg_h/darts_filtered.size()));
	cout << dart_avg << endl;
	rectangle(frame, Point(dart_avg.x, dart_avg.y), Point(dart_avg.x + dart_avg.width, dart_avg.y + dart_avg.height), Scalar( 0, 255, 0 ), 2);

	// cout << "darts post remove: " <<darts.size() << endl;
    //    4. Draw box around faces found
	for( int i = 0; i < darts.size(); i++ ) {
		rectangle(frame2, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar( 0, 255, 0 ), 2);
	}

	for( int i = 0; i < truths.size(); i++) {
		rectangle(frame, Point(truths[i].x, truths[i].y), Point(truths[i].x + truths[i].width, truths[i].y + truths[i].height), Scalar( 0, 0, 255 ), 2);
		rectangle(frame2, Point(truths[i].x, truths[i].y), Point(truths[i].x + truths[i].width, truths[i].y + truths[i].height), Scalar( 0, 0, 255 ), 2);
	}

	imwrite( "detected_darts/"+num+"/detected.jpg", frame2 );
	
	int true_darts = 0;

	for(int t = 0; t < truths.size(); t++) {
		for(int d = 0; d < darts_filtered.size(); d++) {
			if(get_iou(truths[t], darts_filtered[d]) > iou_threshold){
				true_darts++;
				break;
			}
		}
	}

	float tpr = (truths.size() > 0) ? true_darts/truths.size() : 0;

	cout << "image    : " << num << endl;
	cout << "tru darts: " << truths.size() << endl;
       // 3. Print number of Faces found
	cout << "det darts: " << darts_filtered.size() << endl;
	cout << "tpr      : " << tpr << endl;

	float false_pos = darts_filtered.size() - true_darts;
	float false_neg = truths.size() - true_darts;

	cout << "false pos: " << false_pos << endl;
	cout << "false neg: " << false_neg << endl;

	float f1_score = get_f1_score(true_darts, false_pos, false_neg);

	cout << "f1 score : " << f1_score << endl << endl;
}
