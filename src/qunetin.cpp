#include <iostream>
#include <list>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#define PROFILE
#define VAR_KERNEL
#define N_ITER 100

#ifdef PROFILE
#include <time.h>
#include <sys/time.h>
#endif


using namespace std;
using namespace cv;

static constexpr double seuilFond = 3;
static constexpr double N = 10;


double meanIlluminance(const std::list<Mat> &frames, int x, int y) {
    double mean = 0;
    std::for_each(
            frames.cbegin(), frames.cend(),
            [&mean, &x, &y](const Mat &frame) { mean += frame.at<uint8_t>(y, x); }
    );

    return mean / N;
}


double standardDeviationIlluminance(const std::list<Mat> &frames, int x, int y) {
    double mean = meanIlluminance(frames, x, y);
    double ecart_type = 0;

    std::for_each(
            frames.cbegin(), frames.cend(),
            [&mean, &ecart_type, &x, &y](const Mat &frame) {
                ecart_type += (frame.at<uint8_t>(y, x) - mean) * (frame.at<uint8_t>(y, x) - mean);
            }
    );
    return sqrt(ecart_type / N);
}


void computeForeground(const std::list<Mat> &frames, Mat &dst, int cols, int rows) {
    frames.back().copyTo(dst);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (standardDeviationIlluminance(frames, x, y) < seuilFond) {
                dst.at<uint8_t>(y, x) = 0;
            }
            else {
                dst.at<uint8_t>(y, x) = 255;
            }
        }
    }
}


int main() {

/*#ifdef PROFILE
    struct timeval start, end;
    gettimeofday(&start, NULL);
#endif*/
    VideoCapture video = VideoCapture(0);
    if (!video.isOpened()) {
        cout << "The video file cannot be open" << endl;
        return -1;
    }
/*#ifdef PROFILE
    gettimeofday(&end, NULL);
    double e = ((double) end.tv_sec * 1000000.0 + (double) end.tv_usec);
    double s = ((double) start.tv_sec * 1000000.0 + (double) start.tv_usec);
    printf("open video exec time : %lf us\n", (e - s));
#endif*/
    Mat frame, grayFrame, background;
    std::list<Mat> listFrame;
    unsigned char key = '0';
    int i = 0;
    while (key != 'q') {

/*#ifdef PROFILE
        struct timeval start, end;
        gettimeofday(&start, NULL);
#endif*/
        if (!video.read(frame)) {
            std::cout << "end of video" << std::endl;
            break;
        }

        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

/*#ifdef PROFILE
        gettimeofday(&end, NULL);
        double e = ((double) end.tv_sec * 1000000.0 + (double) end.tv_usec);
        double s = ((double) start.tv_sec * 1000000.0 + (double) start.tv_usec);
        printf("lecture frame exec time : %lf us\n", (e - s));
#endif*/
        if (i < N) {
            listFrame.push_back(grayFrame);
            i++;
        }
        else {
            listFrame.pop_front();
            listFrame.push_back(grayFrame.clone());
/*#ifdef PROFILE
            struct timeval start, end;
            gettimeofday(&start, NULL);
#endif*/
            computeForeground(listFrame, background, frame.cols, frame.rows);
/*#ifdef PROFILE
            gettimeofday(&end, NULL);
            double e = ((double) end.tv_sec * 1000000.0 + (double) end.tv_usec);
            double s = ((double) start.tv_sec * 1000000.0 + (double) start.tv_usec);
            printf("calcul arrière plan exec time : %lf us\n", (e - s));
#endif*/
            //Mat kernelRect =  getStructuringElement(MORPH_RECT, Size(1,1));
/*#ifdef PROFILE
            struct timeval start, end;
            gettimeofday(&start, NULL);
#endif*/
            Mat kernelCircle = getStructuringElement(MORPH_ELLIPSE, Size(1,1));
            morphologyEx(background, background, MORPH_OPEN, kernelCircle);
/*#ifdef PROFILE
            gettimeofday(&end, NULL);
            double e = ((double) end.tv_sec * 1000000.0 + (double) end.tv_usec);
            double s = ((double) start.tv_sec * 1000000.0 + (double) start.tv_usec);
            printf("ouverture  exec time : %lf us\n", (e - s));
#endif*/

#ifdef PROFILE
            struct timeval start, end;
            gettimeofday(&start, NULL);
#endif
            vector<vector<Point>> contour;
            findContours(background, contour, RETR_EXTERNAL, CHAIN_APPROX_NONE);
#ifdef PROFILE
            gettimeofday(&end, NULL);
            double e = ((double) end.tv_sec * 1000000.0 + (double) end.tv_usec);
            double s = ((double) start.tv_sec * 1000000.0 + (double) start.tv_usec);
            printf("composant connexe  exec time : %lf us\n", (e - s));
#endif
                    cout << "nombre contour" << contour.size() << endl;
            imshow("Video input", background);
        }

        key = waitKey(5);
    }

    return 0;
}