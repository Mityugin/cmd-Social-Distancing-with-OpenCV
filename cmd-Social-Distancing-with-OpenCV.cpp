#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;

constexpr float CONFIDENCE_THRESHOLD = 0.3; // Confidence threshold
constexpr float NMS_THRESHOLD = 0.4;        // Non-maximum suppression threshold - 0.4
constexpr int NUM_CLASSES = 1;              // Number of classes - 80
constexpr int inpWidth = 608;               // Width of network's input image - 608
constexpr int inpHeight = 608;              // Height of network's input image - 608

// colors for bounding boxes
const Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

int main()
{
    // Only one class for this detection
    vector<string> class_names{ "person" }; 

    // Webcam live stream
    string video = "http://91.223.48.19:557/cgueSCEI?container=mjpeg&stream=main"; 
    VideoCapture source;
    source.open(video.c_str());

    // Path to model files YOLOv4
    auto net = readNetFromDarknet("C:\\TEMP\\Models\\yolov4.cfg", "C:\\TEMP\\Models\\yolov4.weights");

    // Using CUDA
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA_FP16);
    
    auto output_names = net.getUnconnectedOutLayersNames();
    double inference_fps = 0;
    double total_fps = 0;

    cout << "Press any key to exit..." << endl;

    Mat frame, blob;

    // Path to record output video
    string temppath = "C:\\TEMP\\output.mp4";
    VideoWriter writer(temppath, VideoWriter::fourcc('m', 'p', '4', 'v'), 30, Size(1280, 960));

    vector<Mat> detections;
    while (waitKey(1) < 1)
    {

        source >> frame;

        if (frame.empty())
        {
            waitKey(1);
            break;
        }

        auto total_start = chrono::steady_clock::now();

        blobFromImage(frame, blob, 0.00392, Size(inpWidth, inpHeight), Scalar(), true, false, CV_32F);
        net.setInput(blob);

        auto dnn_start = chrono::steady_clock::now();
        net.forward(detections, output_names);
        auto dnn_end = chrono::steady_clock::now();

        vector<int> indices[NUM_CLASSES];
        vector<Rect> boxes[NUM_CLASSES];
        vector<float> scores[NUM_CLASSES];

        for (auto& output : detections)
        {
            const auto num_boxes = output.rows;
            for (int i = 0; i < num_boxes; i++)
            {

                auto x = output.at<float>(i, 0) * frame.cols;
                auto y = output.at<float>(i, 1) * frame.rows;
                auto width = output.at<float>(i, 2) * frame.cols;
                auto height = output.at<float>(i, 3) * frame.rows;
                Rect rect(x - width / 2, y - height / 2, width, height);

                for (int c = 0; c < NUM_CLASSES; c++)
                {
                    auto confidence = *output.ptr<float>(i, 5 + c);
                    if (confidence >= CONFIDENCE_THRESHOLD)
                    {
                        boxes[c].push_back(rect);
                        scores[c].push_back(confidence);
                    }
                }

            }
        }

        for (int c = 0; c < NUM_CLASSES; c++)
            NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

        // Detecting Violators
        vector<int> violators;

        for (size_t i = 0; i < indices[0].size(); ++i)
        {
            auto idx = indices[0][i];
            const auto& rect1 = boxes[0][idx];

            if (indices[0].size() > 1)
            {
                for (size_t j = 0; j < indices[0].size(); ++j)
                {

                    auto jdx = indices[0][j];

                    if (idx != jdx)
                    {
                        const auto& rect2 = boxes[0][jdx];

                        // Distance between pairs of rectangles
                        auto maxheight = max(rect1.height, rect2.height);
                        auto distx = abs((rect1.x + rect1.width / 2) - (rect2.x + rect2.width / 2));
                        auto disty = abs((rect1.y + rect1.height / 2) - (rect2.y + rect2.height / 2));
                        auto dist = sqrt(distx * distx + disty * disty);
                        
                        // If distance < height of rectangle add value to vector
                        if (dist < maxheight)
                        {
                            violators.push_back(idx);
                            violators.push_back(jdx);
                        }
                    }
                }
            }

        }

        // Delete dublicates in violators
        sort(violators.begin(), violators.end());
        violators.erase(unique(violators.begin(), violators.end()), violators.end());


        for (int c = 0; c < NUM_CLASSES; c++)
        {
            for (size_t i = 0; i < indices[c].size(); ++i)
            {
                auto color = colors[c % NUM_COLORS];
                string lbl = class_names[c];

                auto idx = indices[c][i];
                for (auto const& value : violators)
                {
                    // Change color and lable of violator
                    if (idx == value)
                    {
                        color = Scalar(0, 0, 255);
                        lbl = "Violator";
                    }
                }

                const auto& rect = boxes[c][idx];
                rectangle(frame, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), color, 3);

                ostringstream label_ss;
                label_ss << lbl << ": " << fixed << setprecision(2) << scores[c][idx];
                auto label = label_ss.str();

                int baseline;
                auto label_bg_sz = getTextSize(label.c_str(), FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                rectangle(frame, Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), Point(rect.x + label_bg_sz.width, rect.y), color, FILLED);
                putText(frame, label.c_str(), Point(rect.x, rect.y - baseline - 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 0));
            }
        }

        auto total_end = chrono::steady_clock::now();

        inference_fps = 1000.0 / chrono::duration_cast<chrono::milliseconds>(dnn_end - dnn_start).count();
        total_fps = 1000.0 / chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count();
        ostringstream stats_ss;
        stats_ss << fixed << setprecision(2);
        stats_ss << "V-Rus.com/elsa Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
        auto stats = stats_ss.str();

        int baseline;
        auto stats_bg_sz = getTextSize(stats.c_str(), FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        rectangle(frame, Point(0, 0), Point(stats_bg_sz.width, stats_bg_sz.height + 10), Scalar(0, 0, 0), FILLED);
        putText(frame, stats.c_str(), Point(0, stats_bg_sz.height + 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255));

        // Count violators
        ostringstream vio_ss;
        vio_ss << fixed << setprecision(2);
        vio_ss << "Violators: " << violators.size();
        auto vios = vio_ss.str();
        putText(frame, vios.c_str(), Point(20, frame.rows - 50), FONT_HERSHEY_COMPLEX_SMALL, 3, Scalar(0, 0, 255), 2);

        //Write video to file      
        writer << frame;

        namedWindow("output");
        imshow("output", frame);

    }

    // Release source and writer
    source.release();
    writer.release();

    cout << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps << endl;
    cout << "Output file: " + temppath;
    
    while (waitKey(1) < 1)
    {
    }

    // Closes all the frames
    destroyAllWindows();

    return 0;
}