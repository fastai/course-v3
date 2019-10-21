set -ex

OPENCV_VERSION='4.1.0'
CMAKE_OPTS=""
if [ ! -z ${INSTALL_PREFIX} ]; then
  CMAKE_OPTS="-D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}"
fi
APT_PROGRAM="apt-fast"
command -v $APT_PROGRAM > /dev/null 2>&1 || {
  echo "Falling back to apt-get as apt-fast is not installed..."
  APT_PROGRAM="apt-get"
}

sudo $APT_PROGRAM -y update
sudo $APT_PROGRAM install -y build-essential cmake zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libopenexr-dev libgdal-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev libtbb-dev libeigen3-dev python-dev python-tk python-numpy python3-dev python3-tk python3-numpy unzip wget libopenblas-dev libopenblas-base liblapacke-dev libatlas-base-dev liblapack-dev libblas-dev
sudo ln -s /usr/include/lapacke.h /usr/include/x86_64-linux-gnu # corrected path for the library

mkdir -p ~/download
pushd ~/download

wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
unzip -q ${OPENCV_VERSION}.zip
rm ${OPENCV_VERSION}.zip
mv opencv-${OPENCV_VERSION} OpenCV
cd OpenCV
mkdir build
cd build
cmake \
  ${CMAKE_OPTS} \
  -D BUILD_LIST=core,imgproc,imgcodecs \
  -D CMAKE_BUILD_TYPE=Release \
  -D OPENCV_GENERATE_PKGCONFIG=YES \
  -D WITH_CSTRIPES=OFF \
  -D WITH_PTHREADS_PF=OFF \
  -D WITH_QT=OFF \
  -D WITH_OPENGL=OFF \
  -D WITH_OPENCL=OFF \
  -D WITH_OPENMP=OFF \
  -D WITH_TBB=ON \
  -D WITH_GDAL=ON \
  -D WITH_XINE=ON \
  -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF \
  -D ENABLE_PRECOMPILED_HEADERS=OFF \
  -D WITH_IPP=ON \
  -D CPU_BASELINE=NATIVE \
  -D ENABLE_FAST_MATH=ON \
  .. | tee install_cv4.log
make -j $(nproc --all)
sudo make install
sudo ldconfig
popd

echo "Success!"
