#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

#include <sys/stat.h>
#include <unistd.h>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

inline bool exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}


static void getOpticalFlow(const Mat_<float>& flow, string flowpath, vector<float>& minmax)
{
    // double gmin = -71.3109, gmax = 79.274 ; // THESE VALUES WHERE COMPUTED FROM THE FRAMES OF THE FIRST 100 VIDEOS

    double min, max;
    Point min_loc, max_loc ;
    cuda::minMaxLoc(flow, &min, &max, &min_loc, &max_loc);
    minmax.push_back(min) ;
    minmax.push_back(max) ;

    // cout << min << ", " << max << endl ;
    
    // Mat dst_gresc(flow.size(), CV_8UC1) ;
     // dst_gresc.setTo(Scalar::all(0));

    Mat dst_lresc(flow.size(), CV_8UC1) ;
    // dst_lresc.create(flow.size(), CV_8UC1);
    // dst_lresc.setTo(Scalar::all(0));

    for (int y = 0; y < flow.rows; ++y)
    {
        for (int x = 0; x < flow.cols; ++x)
        {
            // Point2f u(flow(y, x), flow(y, x));

            // if (!isFlowCorrect(u))
            // {
                // LOCAL LINEAR RESCALING
            // double new_val = (flow(y, x) - min) / (max - min) ;
            // cout << "--> " << new_val << "   ->   " << (uchar)(255 * new_val) << endl ;
            // cout << "(" << x << ", " << y << ")" << "(" << flow.cols << ", " << flow.rows << ")" << endl ;
            dst_lresc.at<uchar>(y, x) = (uchar)(255 * (flow(y, x) - min) / (max - min));

            // GLOBAL LINEAR RESCALING
            // double val = flow(y, x) ;
            // if (val > gmax)
            //     val = gmax ;
            // else if (val < gmin)
            //     val = gmin ;

            // dst_gresc.at<uchar>(y, x) = (uchar)(255 * (val - gmin) / (gmax - gmin));

                // continue;
            // }
        }
    }
    
    // string ty =  type2str( dst_gresc.type() );
    // printf("Matrix: %s %dx%d \n", ty.c_str(), dst_gresc.cols, dst_gresc.rows );

    // ty =  type2str( dst_lresc.type() );
    // printf("Matrix: %s %dx%d \n", ty.c_str(), dst_lresc.cols, dst_lresc.rows );

    // imwrite(gflowpath, dst_gresc) ;
    imwrite(flowpath, dst_lresc) ;


    // dst_gresc.release() ;
    // dst_lresc.release() ;
    
}

// static void showFlow(const char* name, const GpuMat& d_flow)
static void saveFlow(string flowxpath, string flowypath, vector<float>& xminmax, vector<float>& yminmax, const GpuMat& d_flow)
{
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    // Mat out;
    // drawOpticalFlow(flowx, flowy, out, 10);

    getOpticalFlow(flowx, flowxpath, xminmax) ;
    getOpticalFlow(flowy, flowypath, yminmax) ;

    // imshow(name.c_str(), out);
    // imwrite(name, out) ;
}

int main(int argc, const char* argv[])
{
    cout << argc << endl ;
    if (argc == 1)
    {
        cout << "Binary for computing optical flow using OpenCV and CUDA. Example usage:" << endl ;
        cout << "./compute_optical_flow [file_containing_frame_directories.txt] [file_containing_number_of_frames_per_video.txt]" << endl ;
        cout << "The [file_containing_frame_directories.txt] should have the following structure:" << endl ;
        cout << "/mypath/frames_for_video1" << endl ;
        cout << "/mypath/frames_for_video2" << endl ;
        cout << "..." << endl ;
        cout << "The [file_containing_number_of_frames_per_video.txt] should have the following structure:" << endl ;
        cout << "number_of_frames_for_video1" << endl ;
        cout << "number_of_frames_for_video2" << endl ;
        cout << "..." << endl ;
    }

    string videopaths, numframes ;
    videopaths = argv[1];
    numframes  = argv[2];

    if (argc == 4)
    {
        istringstream iss(argv[3]);
        int device ;
        iss >> device ;
       cuda::setDevice(device) ; 
    }
    // 

    ifstream file(videopaths.c_str());
    ifstream file2(numframes.c_str());

    double gmin=100000, gmax=-100000;
    string line, line2;
    int cnt = 1 ;
    // cout << "New min/max " << gmin << "/" << gmax << endl ;
    while (getline(file, line) && getline(file2, line2))
    {
        // getline(file, line) ;
        // getline(file2, line2) ;
        if (line.empty()) continue;
        istringstream iss(line2);
        int nfr ;
        iss >> nfr ;

        cout << cnt << ") " << line << ", " << nfr << endl ;
        cnt++ ;

        // if (cnt >= 6500)
        //     continue ;

        vector<float> xminmax ;
        vector<float> yminmax ;
        for (int fi = 1 ; fi < nfr ; fi++)
        {

            stringstream ss;
            ss << setw(5) << setfill('0') << fi;
            string fi_str = ss.str();

            stringstream ss2;
            ss2 << setw(5) << setfill('0') << fi+1;
            string fi_str2 = ss2.str();

            string impath1 = line + "/frame_" + fi_str + ".jpg" ;
            string impath2 = line + "/frame_" + fi_str2 + ".jpg" ;

            // string gflowxpath = line + "/gflowx_" + fi_str + ".jpg" ;
            // string gflowypath = line + "/gflowy_" + fi_str + ".jpg" ;
            string flowxpath = line + "/flowx_" + fi_str + ".jpg" ;
            string flowypath = line + "/flowy_" + fi_str + ".jpg" ;

            if (exists(flowypath))
                continue ;

            Mat frame1 = imread(impath1, IMREAD_GRAYSCALE);
            Mat frame2 = imread(impath2, IMREAD_GRAYSCALE);

            if (frame1.empty())
            {
                cerr << "Can't open image ["  << impath1 << "]" << endl;
                return -1;
            }
            if (frame2.empty())
            {
                cerr << "Can't open image ["  << impath2 << "]" << endl;
                return -1;
            }

            if (frame1.size() != frame2.size())
            {
                cerr << "Images should be of equal sizes" << endl;
                return -1;
            }

            // imshow("Frame 0", frame1);
            // imshow("Frame 1", frame2);

            GpuMat d_frame1(frame1);
            GpuMat d_frame2(frame2);

            GpuMat d_flow(frame1.size(), CV_32FC2);

            Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);

            GpuMat d_frame1f;
            GpuMat d_frame2f;

            d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
            d_frame2.convertTo(d_frame2f, CV_32F, 1.0 / 255.0);

            brox->calc(d_frame1f, d_frame2f, d_flow);

            saveFlow(flowxpath, flowypath, xminmax, yminmax, d_flow);

/*            GpuMat planes[2];
            cuda::split(d_flow, planes);

            Mat flowx(planes[0]);
            Mat flowy(planes[1]);

            Point min_loc, max_loc ;
            double min, max ;
            cuda::minMaxLoc(flowx, &min, &max, &min_loc, &max_loc);

            if (min < gmin)
                gmin = min ;

            if (max > gmax)
                gmax = max ;

            */

            // saveFlow(optflowpath, d_flow);

            // return 0 ;

        }
        string xminmax_path = line + "/xminmax.txt" ;
        std::ofstream writeFile ;
        writeFile.open(xminmax_path.c_str());
        for (int i = 0 ; i < xminmax.size() ; i++)
            writeFile << xminmax[i] << endl ;
        writeFile.close() ;

        string yminmax_path = line + "/yminmax.txt" ;
        std::ofstream writeFile2 ;
        writeFile2.open(yminmax_path.c_str());
        for (int i = 0 ; i < yminmax.size() ; i++)
            writeFile2 << yminmax[i] << endl ;
        writeFile2.close() ;
        
        // if (!xminmax.empty())
        //     writeFile.write(reinterpret_cast<char*>(&xminmax[0]),
        //         xminmax.size() * sizeof(xminmax[0]));
        

        // string yminmax_path = line + "/yminmax.txt" ;
        // writeFile.open(yminmax_path.c_str(), std::ios::out);
        // if (!yminmax.empty())
        //     writeFile.write(reinterpret_cast<char*>(&yminmax[0]),
        //         yminmax.size() * sizeof(yminmax[0]));
        // writeFile.close() ;
        
    }

    // cout << "Global min/max " << gmin << "/" << gmax << endl ;

    return 0;
}
