# ONNX
Basic Implementation of ONNX inference using onnx Runtime for Segementation model

Folder Structure 
1. Build --> Has the header,Lib,Bin files of OPENCV 
2.Extra
  2.1. Dependency --> Model(.onnx ) files are stored here , which will be used by the source code and .exe
                     
  2.2 input --> Folder having Input Images.
  2.3 Output_Mask  --> Generated mask for the input image (will be of model output size , but not of input image size )
  2.4 Final_Resized_mask --> Final mask resized into Input Image dimensions 
3.OnnxDep-->has the needed ONNX dependency for executing the code.
4.x64 --> post build folder containing the .exe


Notes::
 We need to manually move the folder into x64/Release , x64/Debug folders , there is script which does automatically , the script is present in Openvino 2023 Repo (https://github.com/Mylavan/Openvino2023.git)
