/*****************************
Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.
********************************/

//#include <aruco/aruco.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;

// dummy prototype for hand moving detection

void createArucoMarker(int id){

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    cv::Mat markerImg;

    cv::aruco::drawMarker(dictionary, id, 200, markerImg);

    imshow("sdf", markerImg);
    imwrite("marker.png" , markerImg);
}

void createCharuco()
{
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 7, 0.04, 0.02, dictionary);

    cv::Mat boardImage;

    board->draw( cv::Size(600, 500), boardImage, 10, 1 );

    cv::imshow("dDf", boardImage);

    imwrite("charuco.png", boardImage);
}

void drawOptFlowMap (const cv::Mat& flow, cv::Mat& cflowmap, int step, const cv::Scalar& color) {

    for(int y = 0; y < cflowmap.rows; y += step)

        for(int x = 0; x < cflowmap.cols; x += step)

        {

            const cv::Point2f& fxy = flow.at< cv::Point2f>(y, x);

            cv::line(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);

            cv::circle(cflowmap, cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 1, color, -1);

        }
}

int decideOrientation(const cv::Mat& flow, cv::Mat& cflow)
{
    int fx = flow.at<Point2f>(0,0).x;
    int fy = flow.at<Point2f>(0,0).y;

    for(int y =0; y < flow.rows; y++)
    {
        for(int x=0; x <  flow.cols; x++)
        {
            fx += flow.at<cv::Point2f>(y,x).x;

            fy += flow.at<cv::Point2f>(y,x).y;
        }
    }
    fx = fx / flow.cols;
    fy = fy / flow.rows;


	return fx;

}


int main()
{

    Size size(160, 120);
    Mat GetImg;
    Mat prvs, next; //current frame

	int curCounter = 0;
	int prevCounter = 0;
	int curValue = 0;
	int flags = 0;
	int temp = 0;

    VideoCapture cap(0);   //0 is the id of video device.0 if you have only one camera
    if(!(cap.read(GetImg))) //get one frame form video
        return 0;

    resize(GetImg, prvs, size);
    cvtColor(prvs, prvs, CV_BGR2GRAY);
    createArucoMarker(1);
    createArucoMarker(13);
    createArucoMarker(28);
    createCharuco();
    //unconditional loop
    while (true) {

        if(!(cap.read(GetImg))) //get one frame form video
            break;
        //Resize
        resize(GetImg, next, size);
        cvtColor(next, next, CV_BGR2GRAY);

        Mat flow;
        calcOpticalFlowFarneback(prvs, next, flow, 0.5, 1, 15, 3, 5, 1.2, 0);
        Mat cflow;
        cvtColor(prvs, cflow, CV_GRAY2BGR);
        drawOptFlowMap(flow, cflow, 10, CV_RGB(0, 255, 0));

        imshow("OpticalFlowFarneback", cflow);

        //Display
        imshow("prvs", prvs);
        imshow("next", next);
        decideOrientation(flow, cflow);
		curCounter += curValue = decideOrientation(flow, cflow);
		waitKey(30);


		if (curValue == 0){
			flags++;
		}

		temp = curCounter - prevCounter;
		if(abs(temp) > 100){
			if (flags >= 5){
				if (temp < 0){
					printf("VLEVO");
					flags = 0;
					prevCounter = curCounter;
				} else {
					printf("VPRAVO");
					flags = 0;
					prevCounter = curCounter;
				}
			}
		}
		else {
			flags = 0;
		}
		printf("cur = %i, prev = %i, curVal = %i, flags = %i  \n", curCounter, prevCounter, curValue, flags);
        prvs = next.clone();
    }
}




