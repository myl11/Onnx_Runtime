
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <array>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include <ctime>
#include <string>
#include <sstream>

using namespace std::chrono;

#define IM_SEGMENTATION_WIDTH			896	// default width of segmented frame
#define IM_SEGMENTATION_HEIGHT			640 // default height of segmented frame
#define IM_SEGMENTATION_CHANNEL			50  // max number of channel with segmented frame
using namespace cv;
using namespace std;
int  PaddingTop = 0;
int PaddingBottom = 0;
int PaddingLeft = 0;
int PaddingRight = 0;
int Original_Input_Height = 0;
int Original_Input_Width = 0;
int outCroppedWidth = 0;
int outCroppedHeight = 0;
int outCroppedOriginY = 0;
int outCroppedOriginX = 0;

static std::vector<float> loadImageandPreProcess(const std::string& filename, int sizeX = IM_SEGMENTATION_WIDTH, int sizeY = IM_SEGMENTATION_HEIGHT)
{
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cout << "No image found.";
    }
    int OriginalInputImageWidth = image.size().width;
    int OriginalInputImageHight = image.size().height;


    cout << "OriginalInputImageWidth : " << OriginalInputImageWidth << endl;
    cout << "OriginalInputImageHight: " << OriginalInputImageHight << endl;
   // imwrite(" input.png ", image);
    //closeup crop calculation
    cv::Rect rect = cv::boundingRect(image);

    outCroppedOriginX = rect.x;
    outCroppedOriginY = rect.y;
    outCroppedWidth = rect.width;
    outCroppedHeight = rect.height;

    cv::Mat croppedImage = image(rect);
    cv::Mat fCroppedImage;
    croppedImage.convertTo(fCroppedImage, CV_32FC1);

    //mean and standard deviation calculation
    cv::Scalar mean, stddev;
    cv::meanStdDev(fCroppedImage, mean, stddev);

    double dMean = mean[0];
    double dStdDev = stddev[0];

    //normalize image pxiel values using image mean & standard deviation
    // using formula : (img – img.mean() / img.std())
    fCroppedImage = (fCroppedImage - dMean) / (dStdDev + 1e-8);

    //old code 
    //cv::Mat fCroppedImageResized;
    //cv::resize(fCroppedImage, fCroppedImageResized, cv::Size(IM_SEGMENTATION_WIDTH, IM_SEGMENTATION_HEIGHT), cv::INTER_NEAREST);

    //New Code 
    int hinput = fCroppedImage.size().height;
    int winput = fCroppedImage.size().width;
    float  aspectRatio = 0;
    int Target_Height = IM_SEGMENTATION_HEIGHT;
    int Target_Width = IM_SEGMENTATION_WIDTH;
    int Resized_Height = 0;
    int Resized_Width = 0;
    //Equal 
    if (winput < hinput)
    {
        aspectRatio = (float)winput / hinput;
        std::cout << aspectRatio << std::endl;
        Resized_Height = Target_Height;
        Resized_Width = (float)aspectRatio * Resized_Height;
        if (Resized_Width > Target_Width)
        {
            Resized_Height = Resized_Height - ((Resized_Width - Target_Width) / aspectRatio);
            Resized_Width = aspectRatio * Resized_Height;
        }

    }
    else
    {
        aspectRatio = (float)hinput / winput;
        Resized_Width = Target_Width;
        Resized_Height = (float)(aspectRatio * Resized_Width);
        if (Resized_Height > Target_Height)
        {
            Resized_Width = Resized_Width - ((Resized_Height - Target_Height) / aspectRatio);
            Resized_Height = aspectRatio * Resized_Width;
        }
    }
    cv::Mat fCroppedImageResized;
    Original_Input_Height = OriginalInputImageHight;
    Original_Input_Width = OriginalInputImageWidth;
    cv::resize(fCroppedImage, fCroppedImageResized, cv::Size(Resized_Width, Resized_Height), cv::INTER_NEAREST);
    //imwrite("Input_Resized.png", fCroppedImageResized);

    int DiffWidth = Target_Width - Resized_Width;
    int DiffHeight = Target_Height - Resized_Height;
    PaddingTop = DiffHeight / 2;
    PaddingBottom = DiffHeight / 2 + DiffHeight % 2;
    PaddingLeft = DiffWidth / 2;
    PaddingRight = DiffWidth / 2 + DiffWidth % 2;

    Mat PaddedImage;
    copyMakeBorder(fCroppedImageResized, PaddedImage, PaddingTop, PaddingBottom, PaddingLeft, PaddingRight, BORDER_CONSTANT, 0);
   // imwrite("Padded_Image.png", PaddedImage);

    std::vector<float> vec;

    int cn = 1;//RGBA , 4 channel
    int iCount = 0;

    const int inputNumChannel = 1;
    const int inputH = IM_SEGMENTATION_HEIGHT;
    const int inputW = IM_SEGMENTATION_WIDTH;

    std::vector<float> vecR;

    uint8_t* pixelPtr = NULL;
    pixelPtr = (uint8_t*)PaddedImage.data;
    vecR.resize(inputH * inputW);


    float b;

    for (int i = 0; i < inputH; i++)
    {
        for (int j = 0; j < inputW; j++)
        {
            float pixelValue = PaddedImage.at<float>(i, j);
            vecR[iCount] = pixelValue;
            iCount++;
        }
    }
    vector<float> input_tensor_values;
    for (auto i = vecR.begin(); i != vecR.end(); ++i)
    {
        input_tensor_values.push_back(*i);
    }

    return input_tensor_values;
}

int main()
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end, start1, end1, outerstart, outerend, outerduration;
    int OptionChoosen = 0;
    while (OptionChoosen == 0)
    {
        std::cout << "Choose From Below Option " << std::endl;
        std::cout << " 1 --> 4CH Autolabelling Model " << std::endl;
        std::cout << " 2 --> 3VT Autobaelling Model " << std::endl;
        std::cout << " 3  -->3VV Autolabelling Model" << std::endl;
        int UserInput = 0;
        cin >> UserInput;
        if (UserInput == 1 || UserInput == 2 || UserInput == 3)
        {
            OptionChoosen = UserInput;
        }
        else
        {
            OptionChoosen = 0;
        }
    }



    vector<String> fn;
    string Image_Name;
    glob("./Extra/input/*.png", fn);
    for (auto f : fn)
    {
        std::cout << "-------------------------------------------NEW FRAME PROCESSING-------------------------------------------" << std::endl;
        outerstart = std::chrono::high_resolution_clock::now();
        // std::cout << f << std::endl;

        string str1 = "./Extra/input";

        // Find first occurrence of "geeks"
        size_t found = f.find(str1);
        /* std::cout << str1.size() << std::endl;*/
        string r = f.substr(str1.size() + 1, f.size());
        r.erase(r.length() - 4);
        // prints the result
        cout << "String is: " << r << std::endl;
        /* cout << "-------------------------------------" << std::endl;*/

        Ort::Env env;
        Ort::RunOptions runOptions;
        Ort::Session session(nullptr);

        constexpr int64_t numChannels = 1;
        int64_t numChannelsoutput;
        constexpr int64_t width = 896;
        constexpr int64_t height = 640;
        constexpr int64_t numClasses = 1000;
        constexpr int64_t numInputElements = numChannels * height * width;


        const std::string imageFile = f;
        auto modelPath = L"./Extra/Dependency/Onnx/OBFetalHeart_AutoLabel_4CH_from_nnUNet_pytorch.onnx";
        auto ModelPath3VT = L"./Extra/Dependency/Onnx/OBFetalHeart_AutoLabel_3VT_from_nnUNet_pytorch.onnx";
        auto ModelPath4CH = L"./Extra/Dependency/Onnx/OBFetalHeart_AutoLabel_4CH_from_nnUNet_pytorch.onnx";
        auto ModelPath3VV = L"./Extra/Dependency/Onnx/OBFetalHeart_AutoLabel_3VV_from_nnUNet_pytorch.onnx";
        if (OptionChoosen == 1)
        {
            modelPath = ModelPath4CH;
            numChannelsoutput = 18;
            std::cout << " CURRENT MODEL LOADED IS ::  4CH " << std::endl;
        }
        else if (OptionChoosen == 2)
        {
            modelPath = ModelPath3VT;
            numChannelsoutput = 6;
            std::cout << " CURRENT MODEL LOADED IS ::  3VT " << std::endl;
        }
        else
        {
            modelPath = ModelPath3VV;
            numChannelsoutput = 9;
            std::cout << " CURRENT MODEL LOADED IS ::  3VV " << std::endl;
        }


        // load image
        std::vector<float> imageVec = loadImageandPreProcess(imageFile);
        if (imageVec.empty()) {
            std::cout << "Failed to load image: " << imageFile << std::endl;
            return 1;
        }

        if (imageVec.size() != numInputElements) {

            std::cout << "Invalid image format. Must be 224x224 RGB image." << std::endl;
            return 1;
        }


        Ort::SessionOptions ort_session_options;

        // create session
        session = Ort::Session(env, modelPath, ort_session_options);

        // Use CPU
        session = Ort::Session(env, modelPath, Ort::SessionOptions{ nullptr });

        //// define shape
        const std::array<int64_t, 4> inputShape = { 1, numChannels, height, width };
        const std::array<int64_t, 4> outputShape = { 1, numChannelsoutput, height, width };

        std::vector<float> results(numChannelsoutput * height * width);
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memory_info, imageVec.data(), imageVec.size(), inputShape.data(), inputShape.size());
        // auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, imageVec.data(), imageVec.size(), inputShape.data(), inputShape.size());
        auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

        // define names
        Ort::AllocatorWithDefaultOptions ort_alloc;
        Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
        Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
        const std::array<const char*, 1> inputNames = { inputName.get() };
        const std::array<const char*, 1> outputNames = { outputName.get() };
        inputName.release();
        outputName.release();

        // run inference
        try {
            session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
        }
        catch (Ort::Exception& e) {
            std::cout << e.what() << std::endl;
            return 1;
        }
        float* floatarr = outputTensor.GetTensorMutableData<float>();

        //Post Processing Code Here 

        int imgSize = IM_SEGMENTATION_WIDTH * IM_SEGMENTATION_HEIGHT;
        unsigned char frameWithMaxPixelValueIndex[IM_SEGMENTATION_WIDTH * IM_SEGMENTATION_HEIGHT];
        memset(frameWithMaxPixelValueIndex, 0, imgSize * sizeof(unsigned char));
        //#pragma omp parallel for
        for (int iPixelIndex = 0; iPixelIndex < imgSize; iPixelIndex++)
        {
            float pixelValue = 0;
            float pixelMaxValue = -INFINITY; //Initialzie max pixel value holder to negative INFINITY
            int   channelIndexWithMaxPixelValue = 0;
            for (int iChannelIndex = 0; iChannelIndex < numChannelsoutput; iChannelIndex++)
            {
                pixelValue = *(floatarr + (iChannelIndex * imgSize + iPixelIndex));
                if (pixelMaxValue < pixelValue)
                {
                    pixelMaxValue = pixelValue;
                    channelIndexWithMaxPixelValue = iChannelIndex;
                }
            }

            frameWithMaxPixelValueIndex[iPixelIndex] = channelIndexWithMaxPixelValue;
        }

        cv::Mat cvframeWithMaxPixelValueIndex = cv::Mat(cv::Size(IM_SEGMENTATION_WIDTH, IM_SEGMENTATION_HEIGHT), CV_8UC1, frameWithMaxPixelValueIndex, cv::Mat::AUTO_STEP);
        cv::imwrite("./Extra/Output_Mask/" + r + ".png", cvframeWithMaxPixelValueIndex);

        outerend = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> outerelapsed_seconds = outerend - outerstart;
        auto duration = duration_cast<seconds>(outerend - outerstart);
        float ms2 = outerelapsed_seconds.count() * 1000.0f;

        std::cout << " Total Time Taken for Infernece for This Image ::" << duration.count() << std::endl;
        //Remove applied padding  back to preprocessed cropped dimension
        cv::Mat preprocess_cvframeWithMaxPixelValueIndex;
        preprocess_cvframeWithMaxPixelValueIndex = cvframeWithMaxPixelValueIndex(Range(PaddingTop, IM_SEGMENTATION_HEIGHT - PaddingBottom), Range(PaddingLeft, IM_SEGMENTATION_WIDTH - PaddingRight));
        //cv::imwrite("frameWithMaxPixelValueIndex_after_preprocess_cropped_resize.png", preprocess_cvframeWithMaxPixelValueIndex);


        //Resize back to Cropped Size 
        cv::Mat FinalResized;
        cv::resize(preprocess_cvframeWithMaxPixelValueIndex, FinalResized, cv::Size(outCroppedWidth, outCroppedHeight), 0, 0, cv::INTER_NEAREST);
        cv::Rect preprocess_cropp_rect;
        preprocess_cropp_rect.x = outCroppedOriginX;
        preprocess_cropp_rect.y = outCroppedOriginY;
        preprocess_cropp_rect.width = outCroppedWidth;
        preprocess_cropp_rect.height = outCroppedHeight;

        //Resize Back to original Input Dimensions 
        cv::Mat maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex = cv::Mat::zeros(cv::Size(Original_Input_Width, Original_Input_Height), CV_8UC1);
        FinalResized.copyTo(maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex(preprocess_cropp_rect));
       // cv::imwrite("frameWithMaxPixelValueIndex_after_maskapplied_orginputimgsize_resize.png", maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex);
        cv::imwrite("./Extra/Final_Resized_mask/" + r + ".png", maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex);



        std::cout << "Myls --> C++ Inference using ONNX Runtime is Done --> Happy :) " << std::endl;


    }

}