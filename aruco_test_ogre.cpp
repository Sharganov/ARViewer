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

#include <iostream>

#include "Ogre.h"
#include <OIS/OIS.h>

#include <aruco/aruco.h>  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <vector>
#include <iostream>

#define MAX_MARKERS 30 // maximun number of markers we assume they will be detected simultaneously

/// Ogre general variables
Ogre::Root* root;
OIS::InputManager* im;
OIS::Keyboard* keyboard;

/// Ogre background variables
Ogre::PixelBox mPixelBox;
Ogre::TexturePtr mTexture;

/// Ogre scene variables
Ogre::SceneNode* ogreNode[MAX_MARKERS];
Ogre::AnimationState *baseAnim[MAX_MARKERS], *topAnim[MAX_MARKERS];

/// ArUco variables
cv::VideoCapture TheVideoCapturer;
cv::Mat newImg, TheInputImageUnd;
aruco::CameraParameters CameraParams, CameraParamsUnd;
float TheMarkerSize=1;
aruco::MarkerDetector TheMarkerDetector;
std::vector<aruco::Marker> TheMarkers;


cv::Mat camMatrix, distCoeffs;


// opencv aruco
cv::Ptr<cv::aruco::Dictionary> dictionary;

int initOgreAR(aruco::CameraParameters camParams, unsigned char* buffer, std::string resourcePath="");
bool readParameters(int argc, char** argv);

static bool readCameraParameters(string filename, cv::Mat &camMatrix, cv::Mat &distCoeffs) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}


void OgreGetPoseParameters(cv::Mat Tvec, cv::Mat Rvec, double position[3], double orientation[4]) throw(cv::Exception) {


    // calculate position vector
    position[0] = -Tvec.ptr< float >(0)[0];
    position[1] = -Tvec.ptr< float >(0)[1];
    position[2] = +Tvec.ptr< float >(0)[2];

    // now calculare orientation quaternion
    cv::Mat Rot(3, 3, CV_32FC1);
    cv::Rodrigues(Rvec, Rot);

    // calculate axes for quaternion
    double stAxes[3][3];
    // x axis
    stAxes[0][0] = -Rot.at< float >(0, 0);
    stAxes[0][1] = -Rot.at< float >(1, 0);
    stAxes[0][2] = +Rot.at< float >(2, 0);
    // y axis
    stAxes[1][0] = -Rot.at< float >(0, 1);
    stAxes[1][1] = -Rot.at< float >(1, 1);
    stAxes[1][2] = +Rot.at< float >(2, 1);
    // for z axis, we use cross product
    stAxes[2][0] = stAxes[0][1] * stAxes[1][2] - stAxes[0][2] * stAxes[1][1];
    stAxes[2][1] = -stAxes[0][0] * stAxes[1][2] + stAxes[0][2] * stAxes[1][0];
    stAxes[2][2] = stAxes[0][0] * stAxes[1][1] - stAxes[0][1] * stAxes[1][0];

    // transposed matrix
    double axes[3][3];
    axes[0][0] = stAxes[0][0];
    axes[1][0] = stAxes[0][1];
    axes[2][0] = stAxes[0][2];

    axes[0][1] = stAxes[1][0];
    axes[1][1] = stAxes[1][1];
    axes[2][1] = stAxes[1][2];

    axes[0][2] = stAxes[2][0];
    axes[1][2] = stAxes[2][1];
    axes[2][2] = stAxes[2][2];

    // Algorithm in Ken Shoemake's article in 1987 SIGGRAPH course notes
    // article "Quaternion Calculus and Fast Animation".
    double fTrace = axes[0][0] + axes[1][1] + axes[2][2];
    double fRoot;

    if (fTrace > 0.0) {
        // |w| > 1/2, may as well choose w > 1/2
        fRoot = sqrt(fTrace + 1.0); // 2w
        orientation[0] = 0.5 * fRoot;
        fRoot = 0.5 / fRoot; // 1/(4w)
        orientation[1] = (axes[2][1] - axes[1][2]) * fRoot;
        orientation[2] = (axes[0][2] - axes[2][0]) * fRoot;
        orientation[3] = (axes[1][0] - axes[0][1]) * fRoot;
    } else {
        // |w| <= 1/2
        static unsigned int s_iNext[3] = {1, 2, 0};
        unsigned int i = 0;
        if (axes[1][1] > axes[0][0])
            i = 1;
        if (axes[2][2] > axes[i][i])
            i = 2;
        unsigned int j = s_iNext[i];
        unsigned int k = s_iNext[j];

        fRoot = sqrt(axes[i][i] - axes[j][j] - axes[k][k] + 1.0);
        double *apkQuat[3] = {&orientation[1], &orientation[2], &orientation[3]};
        *apkQuat[i] = 0.5 * fRoot;
        fRoot = 0.5 / fRoot;
        orientation[0] = (axes[k][j] - axes[j][k]) * fRoot;
        *apkQuat[j] = (axes[j][i] + axes[i][j]) * fRoot;
        *apkQuat[k] = (axes[k][i] + axes[i][k]) * fRoot;
    }
}


void decideOrientation(const cv::Mat& flow, cv::Mat& cflow)
{
    int fx = flow.at<cv::Point2f>(0,0).x;
    int fy = flow.at<cv::Point2f>(0,0).y;

    cv::Point2f fxy = (flow.at<cv::Point2f>(0,0));

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

}

void usage()
{
    cout<<" This program test Ogre version of ArUco (single marker version) \n\n";
    cout<<" Usage <video.avi>|live <camera.yml> <markersize>"<<endl;
    cout<<" <video.avi>|live: specifies a input video file. Use 'live' to capture from camera"<<endl;
    cout<<" <camera.yml>: camera calibration file"<<endl;
    cout<<" <markersize>: in meters "<<endl;
}


int main(int argc, char** argv)
{



    cv::Mat prevImg;
    dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(8));
    /// READ PARAMETERS
    if(!readParameters(argc, argv))
        return false;

    /// CREATE UNDISTORTED CAMERA PARAMS
    CameraParamsUnd=CameraParams;
    CameraParamsUnd.Distorsion=cv::Mat::zeros(4,1,CV_32F);

    /// CAPTURE FIRST FRAME
    TheVideoCapturer.grab();
    TheVideoCapturer.retrieve ( newImg );
    cv::undistort(newImg,TheInputImageUnd,CameraParams.CameraMatrix,CameraParams.Distorsion);

    bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);

    /// INIT OGRE
    initOgreAR(CameraParamsUnd, TheInputImageUnd.ptr<uchar>(0));



    while (TheVideoCapturer.grab())
    {

        /// READ AND UNDISTORT IMAGE
        TheVideoCapturer.retrieve ( newImg );
        cv::undistort(newImg,TheInputImageUnd,CameraParams.CameraMatrix,CameraParams.Distorsion);


        vector< int > ids;
        vector< vector< cv::Point2f > > corners;

        cv::aruco::detectMarkers(newImg, dictionary, corners, ids);

        vector< cv::Mat > rvecs, tvecs;


        //change marker size in meters
        cv::aruco::estimatePoseSingleMarkers(corners, 0.057, CameraParams.CameraMatrix, CameraParams.Distorsion, rvecs, tvecs);


        //throw cv::Exception();
        /// DETECT MARKERS
        TheMarkerDetector.detect(TheInputImageUnd, TheMarkers, CameraParamsUnd, 0.057);


        /// UPDATE BACKGROUND IMAGE
        mTexture->getBuffer()->blitFromMemory(mPixelBox);

        /// UPDATE SCENE
        int i;
        double position[3], orientation[4];
        // show nodes for detected markers
        for(i=0; i<TheMarkers.size() && i<MAX_MARKERS; i++) {

            TheMarkers[i].OgreGetPoseParameters(position, orientation);
            ogreNode[i]->setPosition( position[0], position[1], position[2]  );
            ogreNode[i]->setOrientation( orientation[0], orientation[1], orientation[2], orientation[3]  );
            ogreNode[i]->setVisible(true);
        }

        // hide rest of nodes
        for( ; i<MAX_MARKERS; i++) ogreNode[i]->setVisible(false);

        // Update animation
        double deltaTime = 1.2*root->getTimer()->getMilliseconds()/1000.;
        for(int i=0; i<MAX_MARKERS; i++) {
            baseAnim[i]->addTime(deltaTime);
            topAnim[i]->addTime(deltaTime);
        }

        root->getTimer()->reset();

        /// RENDER FRAME
        if(root->renderOneFrame() == false) break;

        Ogre::WindowEventUtilities::messagePump();

        /// KEYBOARD INPUT
        keyboard->capture();
        if (keyboard->isKeyDown(OIS::KC_ESCAPE)) break;
    }

    im->destroyInputObject(keyboard);
    im->destroyInputSystem(im);
    im = 0;

    delete root;
    return 0;
}


bool readParameters(int argc, char** argv)
{

    if (argc<3) {
        usage();
        return false;
    }
    // read input video
     TheVideoCapturer.open(0);

    if (!TheVideoCapturer.isOpened())
    {
        cerr<<"Could not open video"<<endl;
        return false;
    }

    // read intrinsic file
    try {
        CameraParams.readFromXMLFile(argv[2]);
    } catch (std::exception &ex) {

        cout<<ex.what()<<endl;
        return false;
    }

    if(argc>3) TheMarkerSize=atof(argv[3]);
    else TheMarkerSize=1.;
    return true;
}


/**
  *
  */
int initOgreAR(aruco::CameraParameters camParams, unsigned char* buffer, std::string resourcePath)
{

    /// INIT OGRE FUNCTIONS
    root = new Ogre::Root(resourcePath + "plugins.cfg", resourcePath + "ogre.cfg");
    if (!root->showConfigDialog()) return -1;
    Ogre::SceneManager* smgr = root->createSceneManager(Ogre::ST_GENERIC);


    /// CREATE WINDOW, CAMERA AND VIEWPORT
    Ogre::RenderWindow* window = root->initialise(true);
    Ogre::Camera *camera;
    Ogre::SceneNode* cameraNode;
    camera = smgr->createCamera("camera");
    camera->setNearClipDistance(0.01f);
    camera->setFarClipDistance(10.0f);
    camera->setProjectionType(Ogre::PT_ORTHOGRAPHIC);
    camera->setPosition(0, 0, 0);
    camera->lookAt(0, 0, 1);
    double pMatrix[16];
    camParams.OgreGetProjectionMatrix(camParams.CamSize,camParams.CamSize, pMatrix, 0.05,10, false);
    Ogre::Matrix4 PM(pMatrix[0], pMatrix[1], pMatrix[2] , pMatrix[3],
            pMatrix[4], pMatrix[5], pMatrix[6] , pMatrix[7],
            pMatrix[8], pMatrix[9], pMatrix[10], pMatrix[11],
            pMatrix[12], pMatrix[13], pMatrix[14], pMatrix[15]);
    camera->setCustomProjectionMatrix(true, PM);
    camera->setCustomViewMatrix(true, Ogre::Matrix4::IDENTITY);
    window->addViewport(camera);
    cameraNode = smgr->getRootSceneNode()->createChildSceneNode("cameraNode");
    cameraNode->attachObject(camera);


    /// CREATE BACKGROUND FROM CAMERA IMAGE
    int width = camParams.CamSize.width;
    int height = camParams.CamSize.height;
    // create background camera image
    mPixelBox = Ogre::PixelBox(width, height, 1, Ogre::PF_R8G8B8, buffer);
    // Create Texture
    mTexture = Ogre::TextureManager::getSingleton().createManual("CameraTexture",Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
                                                                 Ogre::TEX_TYPE_2D,width,height,0,Ogre::PF_R8G8B8,Ogre::TU_DYNAMIC_WRITE_ONLY_DISCARDABLE);

    //Create Camera Material
    Ogre::MaterialPtr material = Ogre::MaterialManager::getSingleton().create("CameraMaterial", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
    Ogre::Technique *technique = material->createTechnique();
    technique->createPass();
    material->getTechnique(0)->getPass(0)->setLightingEnabled(false);
    material->getTechnique(0)->getPass(0)->setDepthWriteEnabled(false);
    material->getTechnique(0)->getPass(0)->createTextureUnitState("CameraTexture");

    Ogre::Rectangle2D* rect = new Ogre::Rectangle2D(true);
    rect->setCorners(-1.0, 1.0, 1.0, -1.0);
    rect->setMaterial("CameraMaterial");

    // Render the background before everything else
    rect->setRenderQueueGroup(Ogre::RENDER_QUEUE_BACKGROUND);

    // Hacky, but we need to set the bounding box to something big, use infinite AAB to always stay visible
    Ogre::AxisAlignedBox aabInf;
    aabInf.setInfinite();
    rect->setBoundingBox(aabInf);

    // Attach background to the scene
    Ogre::SceneNode* node = smgr->getRootSceneNode()->createChildSceneNode("Background");
    node->attachObject(rect);


    /// CREATE SIMPLE OGRE SCENE
    // add sinbad.mesh
    Ogre::ResourceGroupManager::getSingleton().addResourceLocation("Sinbad.zip", "Zip", "Popular");
    Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();

    for(int i=0; i<MAX_MARKERS; i++) {

        Ogre::String entityName = "Marker_" + Ogre::StringConverter::toString(i);
        Ogre::Entity* ogreEntity = smgr->createEntity(entityName, "Sinbad.mesh");
        Ogre::Real offset = ogreEntity->getBoundingBox().getHalfSize().y;
        ogreNode[i] = smgr->getRootSceneNode()->createChildSceneNode();
        // add entity to a child node to correct position (this way, entity axis is on feet of sinbad)
        Ogre::SceneNode *ogreNodeChild = ogreNode[i]->createChildSceneNode();
        ogreNodeChild->attachObject(ogreEntity);
        // Sinbad is placed along Y axis, we need to rotate to put it along Z axis so it stands up over the marker
        // first rotate along X axis, then add offset in Z dir so it is over the marker and not in the middle of it
        ogreNodeChild->rotate(Ogre::Vector3(1,0,0), Ogre::Radian(Ogre::Degree(90)));
        ogreNodeChild->translate(0,0,offset,Ogre::Node::TS_PARENT);
        // mesh is too big, rescale!
        const float scale = 0.006675f;
        ogreNode[i]->setScale(scale, scale, scale);

        // Init animation
        ogreEntity->getSkeleton()->setBlendMode(Ogre::ANIMBLEND_CUMULATIVE);
        baseAnim[i] = ogreEntity->getAnimationState("RunBase");
        topAnim[i] = ogreEntity->getAnimationState("RunTop");
        baseAnim[i]->setLoop(true);
        topAnim[i]->setLoop(true);
        baseAnim[i]->setEnabled(true);
        topAnim[i]->setEnabled(true);
    }


    /// KEYBOARD INPUT READING
    size_t windowHnd = 0;
    window->getCustomAttribute("WINDOW", &windowHnd);
    im = OIS::InputManager::createInputSystem(windowHnd);
    keyboard = static_cast<OIS::Keyboard*>(im->createInputObject(OIS::OISKeyboard, true));

    return 1;
}








