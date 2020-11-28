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

using namespace std;
using namespace cv;

/** Function Headers */
vector<string> split( const string &line, char delimiter );
vector<Rect> read_csv(string num);
float get_iou(Rect t, Rect f);
void detectAndDisplay( Mat frame, vector<Rect> truths, string num );
float get_f1_score(float t_p, float f_p, float f_n);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
	string image_n = argv[1];

	Mat frame = imread("source_images/dart"+image_n+".jpg", CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame, read_csv(image_n), image_n );

	// 4. Save Result Image
	imwrite( "detected_faces/t_detected"+image_n+".jpg", frame );

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

	string file_name = "face_truths/"+num+".csv";
	ifstream file(file_name);
	string line;

	while(getline(file, line)) {
		vector<string> tokens = split(line, ',');
		truths.push_back(Rect(stoi(tokens[0]),stoi(tokens[1]),stoi(tokens[2]),stoi(tokens[3])));
	}
	file.close();

	return truths;
}

//https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
float get_iou(Rect t, Rect f) {
	float width = min(f.x + f.width, t.x + t.width) - max(f.x, t.x);
	float height = min(f.y + f.height, t.y + t.height) - max(f.y, t.y);

	float int_area = width * height;
	float uni_area = (f.width * f.height) + (t.width * t.height) - int_area;

	return int_area/uni_area;
}

//https://en.wikipedia.org/wiki/F-score
float get_f1_score(float t_p, float f_p, float f_n) {
	return (t_p == 0 && f_p == 0 && f_n == 0) ? 0 : t_p/(t_p + 0.5 * (f_p+f_n));
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, vector<Rect> truths, string num ) {

	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );


       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

	for( int i = 0; i < truths.size(); i++) {
		rectangle(frame, Point(truths[i].x, truths[i].y), Point(truths[i].x + truths[i].width, truths[i].y + truths[i].height), Scalar( 0, 0, 255 ), 2);
	}

	float iou_threshold = 0.5;
	int true_faces = 0;

	for(int t = 0; t < truths.size(); t++) {
		for(int f = 0; f < faces.size(); f++) {
			if(get_iou(truths[t], faces[f]) > iou_threshold){
				true_faces++;
				break;
			}
		}
	}

	float tpr = (truths.size() > 0) ? true_faces/truths.size() : 0;

	cout << "image: " << num << endl;
	cout << "true faces: " << truths.size() << endl;
       // 3. Print number of Faces found
	cout << "detected faces: " << faces.size() << endl << endl;
	cout << "tpr: " << tpr << endl;

	float false_pos = faces.size() - true_faces;
	float false_neg = truths.size() - true_faces;

	cout << "false positives: " << false_pos << endl;
	cout << "false negatives: " << false_neg << endl << endl;

	float f1_score = get_f1_score(true_faces, false_pos, false_neg);

	cout << "f1 score: " << f1_score << endl << endl << endl;
}