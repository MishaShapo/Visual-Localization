#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>



using namespace cv;
using namespace dnn;


class Detector;
Detector* d;
Detector* createDetector();

void callback(int pos, void*);
	//static const std::string kWinName = "Deep learning object detection in OpenCV";
class BoundingBox {
    public:
        int classId;
        float conf;
        // these two make the top left corner
        int left;
        int top;
        // these two make the bottom right corner
        int right;
        int bottom;

        BoundingBox(int cId, float c, int l, int t, int r, int b) : classId(cId),
                    conf(c), left(l), top(t), right(r), bottom(b) {}
};

class Detector {
    public:
    float confThreshold, nmsThreshold;
    std::vector<std::string> classes;
    float scale;
    bool swapRB;
    int inpWidth;
    int inpHeight;
    std::vector<String> outNames;
    Net net;
    Scalar mean;
    Detector();

    void detect(Mat frame, std::vector<BoundingBox> &bBoxes);
    
    void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, std::vector<BoundingBox> &bBoxes);
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

    std::vector<String> getOutputsNames(const Net& net);

};

