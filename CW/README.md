# COMS30030: Image Processing and Computer Vision Coursework

A dartboard detector based on the Viola-Jones object detector. I use a trained classifier with Viola-Jones and then filter with Hough Space and grouping. 

## Usage

The Project comes with various `.cpp` files which can be used for different detection purposes. This project has been written to work with the `c++11` compiler

### Programs

`faces.cpp` for detecting faces with viola jones.

`face_truth.cpp` for detecting faces and showing Ground Truth values.

`dart_truth.cpp` for detecting dartboards, applying 3 stages of filtering, these are displayed with the Ground Truth values.

### Building

using the provided `makefile` a `.cpp` file can be given as an argument to product a compiled program named `a.out`. The default program compiled with `make` is `faces.cpp`. Note the addition of the `-std=c++11` for compiling the program

### Example build

If you wanted to build the `dart_truth.cpp` program, it can be compiled as follows

```bash
$ make dart_truth
```

### Running

`a.out` can be run from the command line with the image path for the image you want to detect an object in.

### Example Run

There are 16 images that can be tested, `dart0.jpg` to `dart16.jpg`. The programs are given the source folder and the prefix to the image numbering. Say you wanted to run `dart_truth.cpp` on image `dart0.jpg`, this can be compiled and run as follows

```bash
$ make dart_truth
$ ./a.out 0
```
where `0` is the argument passed to the program

### Output

depending on the program that is being run, it will output different files.

`faces.cpp` and `face_truth.cpp` will both output `.jpg` files to the `/detected_faces/` path where each program will output a file named

`detected<num>.jpg`

and

`t_detected<num>.jpg`

respectively where `<num>` is the original image number and the `t_` prefix denotes an image with Ground Truth boxes.

`dart_truth.cpp` outputs various images into the `/detected_darts/<num>/` path where `<num>` is the original image number. In each of these folders you will find an image output for each stage of the detection process

`blurred.jpg` shows a greyscale source image blurred with a gaussian kernel

`x.jpg` `y.jpg` `magnitude.jpg` and `direction.jpg` show the output from the sobel edge detector

`threshold_source.jpg` is the threshold magnitude image which is passed to the hough transform functions

`hough_space_circles.jpg` shows the hough space produced by the circles hough transform function.

`rho_theta_space.jpg` and `threshold_rho_theta.jpg` show the rho theta space for the line hough transform and the threshold image for the same thing.

`hough_space_lines.jpg` and `threshold_lines.jpg` show the hough space for the line hough transform and the threshold image for the same thing.

`detected.jpg` shows the output of the Viola-Jones detector with the dartboard classifier without any filtering. These also show the detected circle centres.

`detected_circles_<num>.jpg` shows the circles detected in the image with `<num>` being the number of circles detected.

`detected_filtered.jpg` is the final output after both hough space filters and grouping has been applied, these still show the circle centres.

### Adding test images

If you wish to add a test image to be detected please add it in the `/source_images/` path following the general naming pattern `dart<num>.jpg` where `<num>` is a new number that has not been used for a previous image.

## Script for easier building and running

included is a `run.sh` script file that makes the process of building and running these programs easier.

### Usage

`run.sh` takes one argument with an additional optional argument depending on which program you intend to run (since technically you don't need to specify a program if you wish to build and run `faces.cpp`)

The first argument is the program name you wish to build and the second is the image you want to detect. An argument of `all` has been created for ease of running the program on all test images.

The script will first build the program with `make` before running the program with `./a.out` on the image specified or on all 16 test images (Note if you have added a test image these will not be included in the `all` argument for the script, however the script will run the new image on its own).

### Example usage

If you wanted to build and run `dart_truth.cpp` on image `dart4.jpg`

```bash
$ ./run.sh dart_truth 4
```

If you wanted to build and run `face_truth.cpp` on all test images

```bash
$ ./run.sh face_truth all
```
