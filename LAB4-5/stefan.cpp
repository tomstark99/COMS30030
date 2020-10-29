// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
using namespace cv;
 
//function for the sobel filter
void sobel(cv::Mat &input, int size,    
           cv::Mat &xDerivative,
           cv::Mat &yDerivative,
           cv::Mat &gradientMagnitude,
           cv::Mat &directionGradient);
 
void sobel(cv::Mat &input,cv::Mat &xDerivative,cv::Mat &yDerivative,cv::Mat &gradientMagnitude,cv::Mat &directionGradient){
    //declare convolution matrixes for x and y
    Mat sobelX, sobelY;
    // 3 x 3 - x direction
    getDerivKernels(sobelX, sobelY, 0, 1, 3, false, CV_32F);
    cv::Mat kernelX = sobelX * sobelY.t();
 
    // 3 x 3 - y direction
    getDerivKernels(sobelX, sobelY, 1, 0, 3, false, CV_32F);
    cv::Mat kernelY = sobelX * sobelY.t();
 
    int kernelRadiusX = (kernelX.size[1] - 1) / 2;
 
    int kernelRadiusY = (kernelX.size[1] - 1) / 2;
    for (int i = 0; i < input.rows; i++)
    {
        for(int j=0; j< input.cols; j++) {
        double sum_X = 0.0;
        double sum_Y = 0.0;
        for (int m = -kernelRadiusX; m <= kernelRadiusX; m++)
        {
            for (int n = -kernelRadiusY; n <= kernelRadiusY; n++)
            {
                // find the correct indices we are using
                int imagex = i + m + kernelRadiusX;
                int imagey = j + n + kernelRadiusY;
                int kernelx = m + kernelRadiusX;
                int kernely = n + kernelRadiusY;
 
                // get the values from the padded image and the kernel
                int imageval = (int)input.at<uchar>(imagex, imagey);
                double kernalvalX = kernelX.at<double>(kernelx, kernely);
                double kernelvalY = kernelY.at<double>(kernelx, kernely);
 
                // do the multiplication
                sum_X += imageval * kernalvalX;
                sum_Y += imageval * kernelvalY;
            }
        }
        // set the output value as the sum of the convolution
        xDerivative.at<uchar>(i, j) = (uchar)sum_X;
        yDerivative.at<uchar>(i, j) = (uchar)sum_Y;
        gradientMagnitude.at<uchar>(i, j) = (uchar)sqrt((sum_X * sum_Y) + (sum_X * sum_Y));
        directionGradient.at<uchar>(i, j) = (uchar)atan2(sum_Y, sum_X);
    }
    }
    imwrite("coin_x.jpg", xDerivative);
    imwrite("coin_y.jpg", yDerivative);
    imwrite("coin_mag.jpg", gradientMagnitude);
    imwrite("coin_dir.jpg", directionGradient);
}
 
void ThresholdImage(cv::Mat &input, float threshold) {
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            if(input.at<uchar>(i,j) > threshold) 
                input.at<uchar>(i,j) = 255;
            else
                input.at<uchar>(i,j) = 0;
        }
    }       
}
 
vector<Point3d> HoughTransform(cv::Mat &input, cv::Mat &gradient_dir, float threshold)
{
    ThresholdImage(gradient_dir, threshold);
    int rLen = input.rows / 2;
    int dims[] = {input.rows, input.cols, rLen};
    Mat houghSpace(3, dims, CV_8UC(1), Scalar::all(0));
    for (int y = 0; y < input.rows; y++)
    {
        for (int x = 0; x < input.cols; x++)
        {
            for (int r = 0; r < rLen; r++)
            {
                houghSpace.at<cv::Vec3i>(y, x)[r]=0;
            }
        }
    }
 
    for (int y = 0; y < input.rows; y++){
        for (int x = 0; x < input.cols; x++){
            for (int r = 20; r < rLen; r++){
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
    vector<Point3d> circles;
    for (int y = 0; y < input.rows; y++){
        for (int x = 0; x < input.cols; x++){
            for (int r = 20; r < rLen; r++){
                if (houghSpace.at<cv::Vec3i>(y, x)[r] > 100){
                    Point3d temp(x, y, r);
                    circles.push_back(temp);
                    //circle(input, Point(x, y), r, (255, 0, 0), 5);
                }
            }
        }
    }
    return circles;
}
int main(int argc, char **argv) {
 
    // LOADING THE IMAGE
     char *imageName = argv[1];
 
    Mat image;
    image = imread(imageName, 1);
 
    if (argc != 2 || !image.data){
        printf(" No image data \n ");
        return -1;
    }
 
    // Convert the image into Grey Scale
    Mat gray_image;
    cvtColor(image, gray_image, CV_BGR2GRAY);
    imwrite("greyScaleCoin.jpg", gray_image);
 
    Mat xDerivative, yDerivative, gradientMagnitude, directionGradient;
    Mat input;
    copyMakeBorder(gray_image, input, 1, 1, 1, 1, BORDER_REPLICATE);
    // For the kernels
    xDerivative.create(input.size(), input.type());
    yDerivative.create(input.size(), input.type());
    gradientMagnitude.create(input.size(), input.type());
    directionGradient.create(input.size(), input.type());
 
    sobel(input, xDerivative, yDerivative, gradientMagnitude, directionGradient);
    vector<Point3d> circles = HoughTransform(image, directionGradient, 100);
    std::cout << circles.size();
    imwrite("coin_mag3.jpg", image);
 
    return 0;
}
