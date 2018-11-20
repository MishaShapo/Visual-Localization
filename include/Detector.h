#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>



using namespace cv;
using namespace dnn;



	//static const std::string kWinName = "Deep learning object detection in OpenCV";
class BoundingBox {
    public:
        std::string className;
        // these two make the top left corner
        int left;
        int top;
        // these two make the bottom right corner
        int right;
        int bottom;

        BoundingBox(std::string c, int l, int t, int r, int b) : className(c), left(l), top(t), right(r), bottom(b) {}
};


class Detector {
    private:

    public:
    
        Detector();
        void detect(Mat frame, std::vector<BoundingBox> &bBoxes);
    


};

