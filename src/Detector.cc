#include "arapaho.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include "opencv2/core/core.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include "Detector.h"


#define MAX_OBJECTS_PER_FRAME (100)
#define TARGET_SHOW_FPS (10)

#define _ENABLE_OPENCV_SCALING

using namespace cv;





static char modelPath[] = "input.weights";
static char configPath[] = "input.cfg";
static char dataPath[] = "input.data";

static ArapahoV2* p;


    Detector::Detector() {
        std::cout << "Detector constructor" << std::endl;





        
        p = new ArapahoV2();
        if(!p) {
            EPRINTF("Setup failed!\n");
            return;
        }
        ArapahoV2Params ap;
        ap.datacfg = dataPath;
        ap.cfgfile = configPath;
        ap.weightfile = modelPath;
        ap.nms = 0.4;
        ap.maxClasses = 2;

        int expectedW = 0;
        int expectedH = 0;
        bool ret = false;
        ret = p->Setup(ap, expectedW, expectedH);
        if(false == ret) {
            EPRINTF("Setup failed!\n");
            std::cout << "setup failed" << std::endl;
            if(p) delete p;
            p = 0;
            return;
        }



    }

    void Detector::detect(Mat frame, std::vector<BoundingBox> &bBoxes) {
        
        if(frame.empty()) {
            std::cout << "WARNING: detect recieved empty frame. Might be an issue" << std::endl;
        }
	    ArapahoV2ImageBuff arapahoImage;
	    int imageWidth = frame.size().width;
	    int imageHeight = frame.size().height;

	    arapahoImage.bgr = frame.data;
	    arapahoImage.w = imageWidth;
	    arapahoImage.h = imageHeight;
	    arapahoImage.channels = 3;

	    int numObjects = 0;

	    #ifdef _ENABLE_OPENCV_SCALING
        std::cout<< "using frame " << std::endl;
        
	    p->Detect(frame, .24, .5, numObjects);
	    #else
        std::cout<< "using ArapahoV2ImageBuff" << std::endl;
	    p->Detect(arapahoImage, .24, .5, numObjects);
	    #endif
        printf("==> Detected [%d] objects\n", numObjects);
	    box* boxes = 0;
        std::string* labels;
	    if(numObjects > 0 && numObjects < MAX_OBJECTS_PER_FRAME) {
            boxes = new box[numObjects];
            labels = new std::string[numObjects];
            std::cout << "let's see how many objects: " << numObjects  << std::endl;
            if(!boxes) {
	            if(p) delete p;
	            p = 0;
	            return;
                std::cout << "returning because no boxes" << std::endl;
            }
            if(!labels) {
	            if(p) delete p;
                p = 0;
                if(boxes) {
                    delete[] boxes;
                    boxes = NULL;
                }
                std::cout << "returning because no labels" << std::endl;
                return;
            }

            p->GetBoxes(boxes, labels, numObjects);

            int objId = 0;
	        int leftTopX = 0, leftTopY = 0, rightBotX = 0,rightBotY = 0;

            for (objId = 0; objId < numObjects; objId++) {
                leftTopX = 1 + imageWidth*(boxes[objId].x - boxes[objId].w / 2);
                leftTopY = 1 + imageHeight*(boxes[objId].y - boxes[objId].h / 2);

                rightBotX = 1 + imageWidth*(boxes[objId].x + boxes[objId].w / 2);
                rightBotY = 1 + imageHeight*(boxes[objId].y + boxes[objId].h / 2);

                // Show image and overlay using OpenCV
                rectangle(frame, cvPoint(leftTopX, leftTopY), cvPoint(rightBotX, rightBotY), CV_RGB(255, 0, 0), 1, 8, 0);
                bBoxes.push_back(BoundingBox(labels[objId], leftTopX, leftTopY, rightBotX, rightBotY));
            }

            if (boxes) {
                delete[] boxes;
                boxes = NULL;
            }

            if (labels) {
                delete[] labels;
                labels = NULL;
            }
        }



    }
    



/*
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
*/
