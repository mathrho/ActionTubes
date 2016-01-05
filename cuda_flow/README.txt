
CUDA_FLOW
***************************************************************************************************************************
***************************************************************************************************************************

This code shows several examples for using the dense optical flow methods from the GPU module of the OpenCV Library.
The version of the OpenCV library should be equal or higher than 2.4.
You should have a NVIDA graphic card and the OpenCV library compiled with CUDA support to run the examples.

***************************************************************************************************************************
***************************************************************************************************************************

Compiling the program
---------------------------------------------------------------------------------------------------------------------------
In the program folder please type:
cmake .
make

After compiling and linking you should have three executables:
brox_flox
farneback_flow
lk_flow (this method seems to have some bugs in OpenCV 2.4.3)

***************************************************************************************************************************
***************************************************************************************************************************

Running the program
---------------------------------------------------------------------------------------------------------------------------
The program accepts command line arguments.
To run each of the methods you should pass the path of a video file.
Examples:

./brox_flow video.avi
./farneback_flow video.avi
./lk_flow video.avi

***************************************************************************************************************************
***************************************************************************************************************************

More Info
---------------------------------------------------------------------------------------------------------------------------
Name: cuda_flow
Author: Pablo F. Alcantarilla
Date: 26 / 11 / 2012

For further questions, please contact me at:
Contact: pablofdezalc@gmail.com

Thanks to my colleague Alireza Fathi for choosing the parameters of Brox method.


