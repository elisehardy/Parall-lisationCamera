#include <iostream>
#include <list>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


#ifdef PROFILING

#include <sys/time.h>


static uint32_t tab = 0;

/**
 * Compute the number of nanoseconds taken to execute 'code'.
 */
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
             printf("\t-> done in : %ld ms\n", e - s);\
    })
#else
#define PROFILE(code, description) ({\
        code;\
    })
#endif

static constexpr double AREA_SIZE_HUMAN = 15000;
static constexpr double BACKGROUND_THRESHOLD = 5;
static constexpr double BACKGROUND_THRESHOLD_SQUARED = BACKGROUND_THRESHOLD * BACKGROUND_THRESHOLD;
static constexpr double MINIMUM_FRAME_COUNT = 5;
static constexpr double INV_MINIMUM_FRAME_COUNT = 1./5;


static double meanIlluminance(const std::list<cv::Mat> &frames, int32_t x, int32_t y) {
    double mean = 0;
    
    for (const cv::Mat &frame : frames) {
        mean += frame.at<uint8_t>(y, x);
    }
    mean *= INV_MINIMUM_FRAME_COUNT;
    
    return mean;
}


static double standardDeviationIlluminance(const std::list<cv::Mat> &frames, int32_t x, int32_t y) {
    double mean = meanIlluminance(frames, x, y);
    double sd = 0, tmp;
    
    for (const cv::Mat &frame : frames) {
        tmp = frame.at<uint8_t>(y, x) - mean;
        sd += tmp * tmp;
    }
    
    return sd * INV_MINIMUM_FRAME_COUNT;
}


static void computeForeground(const std::list<cv::Mat> &frames, cv::Mat &dst, int32_t cols, int32_t rows) {
    frames.back().copyTo(dst);
    
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (standardDeviationIlluminance(frames, x, y) < BACKGROUND_THRESHOLD_SQUARED) {
                dst.at<uint8_t>(y, x) = 0;
            }
            else {
                dst.at<uint8_t>(y, x) = 255;
            }
        }
    }
}


int main() {
    cv::Mat drawn, frame, grayFrame, background, kernelCircle, labels, stats, centroids;
    int32_t frameCount = 0, humans, connectedCount;
    std::vector<std::vector<cv::Point>> contour;
    std::vector<cv::Rect> boundingBoxes;
    std::list<cv::Mat> listFrame;
    unsigned char key = '0';
    cv::VideoCapture video;
    
    PROFILE(
            video = cv::VideoCapture(0);
            if (!video.isOpened()) {
                std::cout << "The video file cannot be open" << std::endl;
                return -1;
            },
            "Opening video..."
    );
    
    kernelCircle = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    
    while (key != 'q') {
        
        #ifdef PROFILING
        printf("\n==============================\n");
        #endif
        
        PROFILE(
                if (!video.read(frame)) {
                    std::cout << "end of video" << std::endl;
                    break;
                }
                cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY),
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
                morphologyEx(background, background, cv::MORPH_ERODE, kernelCircle);
                morphologyEx(background, background, cv::MORPH_DILATE, kernelCircle),
                "Applying morphologic operations..."
        );
        
        PROFILE(
                connectedCount = connectedComponentsWithStats(background, labels, stats, centroids);
                humans = connectedCount - 1;
                boundingBoxes.clear();
                for (int32_t i = 1; i <= connectedCount; i++) {
                    if (stats.at<int32_t>(i - 1, cv::CC_STAT_AREA) < AREA_SIZE_HUMAN) {
                        humans--;
                    }
                    else {
                        boundingBoxes.emplace_back(
                                stats.at<int32_t>(i - 1, cv::CC_STAT_LEFT),
                                stats.at<int32_t>(i - 1, cv::CC_STAT_TOP),
                                stats.at<int32_t>(i - 1, cv::CC_STAT_WIDTH),
                                stats.at<int32_t>(i - 1, cv::CC_STAT_HEIGHT)
                        );
                    }
                },
                "Finding connected components..."
        );
        
        PROFILE(
                drawn = frame.clone();
                for (auto &e : boundingBoxes) {
                    rectangle(drawn, e.tl(), e.br(), { 0, 0, 255 }, 2);
                },
                "Computing output..."
        );
        
        PROFILE(
                imshow("Video input", drawn),
                "Displaying result..."
        );
        
        key = static_cast<uint8_t>(cv::waitKey(5));
    }
    
    return 0;
}
