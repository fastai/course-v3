# SwiftCV

Minimal Swift for TensorFlow OpenCV4 bindings, partially ported from [gocv](https://github.com/hybridgroup/gocv).

OpenCV Functions exposed:
 * resize
 * getRotationMatrix2D
 * warpAffine
 * copyMakeBorder
 * GaussianBlur
 * remap
 * imdecode
 * imread
 * cvtColor
 * flip
 * transpose
 
OpenCV's `Mat` can be converted to S4TF's `Tensor` and `ShapedArray` types. 
 
See `Extra/Test.ipynb` and `Tests` as an example of usage.

This notebook was executed using [updated swift-jupyter docker image](https://github.com/vvmnnnkv/swift-jupyter/tree/opencv4) with OpenCV4 installed.   

## Usage
Include as SwiftPM package:

`.package(url: "https://github.com/vvmnnnkv/SwiftCV.git", .branch("master"))`

NOTE: OpenCV4 must installed in order for package to compile.


## Disclaimer
Currently this package is just an example of OpenCV/S4TF integration with no safety checks and guarantees to work properly :)

## License
OpenCV C API, (c) Copyright [gocv](https://github.com/hybridgroup/gocv) authors. 