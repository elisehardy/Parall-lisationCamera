#include <iostream>
#include <list>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


#ifdef PROFILING

#include <sys/time.h>


static uint32_t tab = 0;

#define PROFILE(code, description) ({\
             for (uint32_t _profiler_i = 0; _profiler_i < tab; _profiler_i++) {printf("\t");}\
             printf("%s\n", description);\
             struct timeval _profiler_start{}, _profiler_end{};\
             gettimeofday(&_profiler_start, nullptr);\
             tab++;\
             code;\
             tab--;\
             gettimeofday(&_profiler_end, nullptr);\
             int64_t e = _profiler_end.tv_sec * 1000000l + _profiler_end.tv_usec;\
             int64_t s = _profiler_start.tv_sec * 1000000l + _profiler_start.tv_usec;\
             for (uint32_t _profiler_i = 0; _profiler_i < tab; _profiler_i++) {printf("\t");}\
             printf("\t-> done in : %ld us\n", e - s);\
    })
#else
#define PROFILE(code, description) ({\
        code;\
    })
#endif

using namespace std;
using namespace cv;

static constexpr double AREA_SIZE_HUMAN = 30000;
static constexpr double BACKGROUND_THRESHOLD = 4;
static constexpr double MINIMUM_FRAME_COUNT = 3;


static double meanIlluminance(const std::list<Mat> &frames, int x, int y) {
    double mean = 0;
    
    std::for_each(
            frames.cbegin(), frames.cend(),
            [&mean, &x, &y](const Mat &frame) { mean += frame.at<uint8_t>(y, x); }
    );
    mean /= MINIMUM_FRAME_COUNT;
    
    return mean;
}


static double standardDeviationIlluminance(const std::list<Mat> &frames, int x, int y) {
    double mean = meanIlluminance(frames, x, y);
    double ecart_type = 0;
    double sd;
    
    std::for_each(
            frames.cbegin(), frames.cend(),
            [&mean, &ecart_type, &x, &y](const Mat &frame) {
                ecart_type += (frame.at<uint8_t>(y, x) - mean) * (frame.at<uint8_t>(y, x) - mean);
            }
    );
    sd = sqrt(ecart_type / MINIMUM_FRAME_COUNT);
    
    return sd;
}


static void computeForeground(const std::list<Mat> &frames, Mat &dst, int cols, int rows) {
    frames.back().copyTo(dst);
    
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (standardDeviationIlluminance(frames, x, y) < BACKGROUND_THRESHOLD) {
                dst.at<uint8_t>(y, x) = 0;
            }
            else {
                dst.at<uint8_t>(y, x) = 255;
            }
        }
    }
}


int main() {
    Mat frame, grayFrame, background, kernelCircle, labels, stats, centroids;
    int frameCount = 0, humans, connectedCount;
    vector<vector<Point>> contour;
    std::list<Mat> listFrame;
    unsigned char key = '0';
    VideoCapture video;
    
    PROFILE(
            video = VideoCapture(0);
            if (!video.isOpened()) {
                cout << "The video file cannot be open" << endl;
                return -1;
            },
            "Opening video..."
    );
    
    kernelCircle = kernelCircle = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
    
    while (key != 'q') {
        
        #ifdef PROFILING
        printf("\n==============================\n");
        #endif
        
        PROFILE(
                if (!video.read(frame)) {
                    std::cout << "end of video" << std::endl;
                    break;
                }
                cvtColor(frame, grayFrame, COLOR_BGR2GRAY),
                "Reading frame..."
        );
        
        if (frameCount < MINIMUM_FRAME_COUNT) { // Need at least MINIMUM_FRAME_COUNT frames
            listFrame.push_back(grayFrame);
            frameCount++;
            continue;
        }
        
        listFrame.pop_front();
        listFrame.push_back(grayFrame.clone());
        
        PROFILE(
                computeForeground(listFrame, background, frame.cols, frame.rows),
                "Computing foreground..."
        );
        
        PROFILE(
                morphologyEx(background, background, MORPH_ERODE, kernelCircle);
                morphologyEx(background, background, MORPH_DILATE, kernelCircle),
                "Applying morphologic operations..."
        );

        PROFILE(
                connectedCount = connectedComponentsWithStats(background, labels, stats, centroids);
                humans = connectedCount - 1;
                for(int32_t i = 1; i <= connectedCount; i++ ) {
                    if (stats.at<int>(i - 1, cv::CC_STAT_AREA) < AREA_SIZE_HUMAN) {
                        humans--;
                    }
                },
                "Finding connected components..."
        );

        cout << "Nombre d'humain(s): " << humans << endl;
        
        PROFILE(
                imshow("Video input", background),
                "Displaying result..."
        );
        
        key = static_cast<unsigned char>(waitKey(5));
    }
    
    return 0;
}
