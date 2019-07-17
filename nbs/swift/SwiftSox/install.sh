cd ~/download/
wget https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz/download
mv download sox-14.4.2.tar.gz
tar xf sox-14.4.2.tar.gz
cd sox-14.4.2/
sudo apt install libopencore-amrnb-dev libopencore-amrwb-dev libao-dev libflac-dev libmp3lame-dev libtwolame-dev libltdl-dev libmad0-dev libid3tag0-dev libvorbis-dev libpng-dev libsndfile1-dev libwavpack-dev autoconf automake
autoreconf -i
./configure
make -j
make -s
sudo make install

