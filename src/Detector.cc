#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "Detector.h"


using namespace cv;
using namespace dnn;




	//static const std::string kWinName = "Deep learning object detection in OpenCV";

    Detector::Detector() {
	    confThreshold = .4;
	    nmsThreshold = .3;
	    scale = .00394;
	    mean = Scalar(0,0,0);
	    swapRB = 1;
	    inpWidth = 640;
	    inpHeight = 480;

	    std::string modelPath = "yolov2-tiny.weights";
	    std::string configPath = "cfg/yolov2-tiny.cfg";

	    // Open file with classes names.


	    std::string file = "data/coco.names";
	    std::ifstream ifs(file.c_str());
	    if (!ifs.is_open())
	        CV_Error(Error::StsError, "File " + file + " not found");
	    std::string line;
	    while (std::getline(ifs, line))
	    {
	        classes.push_back(line);
	    }


	    // Load a model.
	    net = readNetFromDarknet(configPath, modelPath);
	    //net = readNetFromDarknet(modelPath, configPath);
	    net.setPreferableBackend(0);
	    net.setPreferableTarget(DNN_TARGET_OPENCL);
	    outNames = getOutputsNames(net);

	    // Create a window
	    //startWindowThread();
	    //namedWindow("test", WINDOW_AUTOSIZE);
	    //int initialConf = (int)(confThreshold * 100);
	    //createTrackbar("Confidence threshold, %", "test", &initialConf, 99, callback);

    }

    void Detector::detect(Mat frame, std::vector<BoundingBox> &bBoxes) {
	        // Process frames.
	    Mat blob;

	    // Create a 4D blob from a frame.
	    Size inpSize(inpWidth > 0 ? inpWidth : frame.cols,
		         inpHeight > 0 ? inpHeight : frame.rows);
	    //blobFromImage(frame, blob, scale, inpSize, mean, swapRB, false);
	    blob = blobFromImage(frame, scale, inpSize, mean, swapRB, false);

	    // Run a model.
	    net.setInput(blob);
	    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	    {
	        resize(frame, frame, inpSize);
	        Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
	        net.setInput(imInfo, "im_info");
	    }
	    std::vector<Mat> outs;
	    net.forward(outs, outNames);

	    postprocess(frame, outs, net, bBoxes);

	    // Put efficiency information.
	    //std::vector<double> layersTimes;
	    //double freq = getTickFrequency() / 1000;
	    //double t = net.getPerfProfile(layersTimes) / freq;
	    //std::string label = format("Inference time: %.2f ms", t);
	    //putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
	
	    //waitKey(1);
	    //imshow("test", frame);
	    //waitKey(0);
	    //destroyAllWindows();
    }
    




    void Detector::postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, std::vector<BoundingBox> &bBoxes) {
        static std::vector<int> outLayers = net.getUnconnectedOutLayers();
        static std::string outLayerType = net.getLayer(outLayers[0])->type;

        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<Rect> boxes;
        if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
        {
            // Network produces output blob with a shape 1x1xNx7 where N is a number of
            // detections and an every detection is a vector of values
            // [batchId, classId, confidence, left, top, right, bottom]
            CV_Assert(outs.size() == 1);
            float* data = (float*)outs[0].data;
            for (size_t i = 0; i < outs[0].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left = (int)data[i + 3];
                    int top = (int)data[i + 4];
                    int right = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width = right - left + 1;
                    int height = bottom - top + 1;
                    classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                    boxes.push_back(Rect(left, top, width, height));
                    confidences.push_back(confidence);
                }
            }
        }
        else if (outLayerType == "DetectionOutput")
        {
            // Network produces output blob with a shape 1x1xNx7 where N is a number of
            // detections and an every detection is a vector of values
            // [batchId, classId, confidence, left, top, right, bottom]
            CV_Assert(outs.size() == 1);
            float* data = (float*)outs[0].data;
            for (size_t i = 0; i < outs[0].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left = (int)(data[i + 3] * frame.cols);
                    int top = (int)(data[i + 4] * frame.rows);
                    int right = (int)(data[i + 5] * frame.cols);
                    int bottom = (int)(data[i + 6] * frame.rows);
                    int width = right - left + 1;
                    int height = bottom - top + 1;
                    classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                    boxes.push_back(Rect(left, top, width, height));
                    confidences.push_back(confidence);
                }
            }
        }
        else if (outLayerType == "Region")
        {
            for (size_t i = 0; i < outs.size(); ++i)
            {
                // Network produces output blob with a shape NxC where N is a number of
                // detected objects and C is a number of classes + 4 where the first 4
                // numbers are [center_x, center_y, width, height]
                float* data = (float*)outs[i].data;
                for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
                {
                    Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                    Point classIdPoint;
                    double confidence;
                    minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                    if (confidence > confThreshold)
                    {
                        int centerX = (int)(data[0] * frame.cols);
                        int centerY = (int)(data[1] * frame.rows);
                        int width = (int)(data[2] * frame.cols);
                        int height = (int)(data[3] * frame.rows);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;

                        classIds.push_back(classIdPoint.x);
                        confidences.push_back((float)confidence);
                        boxes.push_back(Rect(left, top, width, height));
                    }
                }
            }
        }
        else
            CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

        std::vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            int idx = indices[i];
            Rect box = boxes[idx];
            drawPred(classIds[idx], confidences[idx], box.x, box.y,
                     box.x + box.width, box.y + box.height, frame);
            bBoxes.push_back(BoundingBox(classIds[idx], confidences[idx], box.x,
                                        box.y, box.x + box.width, box.y + box.height));
        }
    }

    void Detector::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
    {
        rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

        std::string label = format("%.2f", conf);
        if (!classes.empty())
        {
            CV_Assert(classId < (int)classes.size());
            label = classes[classId] + ": " + label;
        }

        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        top = max(top, labelSize.height);
        rectangle(frame, Point(left, top - labelSize.height),
                  Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
        putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
    }


    std::vector<String> Detector::getOutputsNames(const Net& net)
    {
        static std::vector<String> names;
        if (names.empty())
        {
            std::vector<int> outLayers = net.getUnconnectedOutLayers();
            std::vector<String> layersNames = net.getLayerNames();
            names.resize(outLayers.size());
            for (size_t i = 0; i < outLayers.size(); ++i)
                names[i] = layersNames[outLayers[i] - 1];
        }
        return names;
    }



