pushd ~/git
sudo apt-fast install git build-essential libxml2-dev libfftw3-dev libmagickwand-dev libopenexr-dev liborc-0.4-0 gobject-introspection libgsf-1-dev libglib2.0-dev liborc-0.4-dev python-gi-dev libgirepository1.0-dev automake libtool swig gtk-doc-tools libopenslide-dev libmatio-dev libgif-dev libwebp-dev
git clone https://github.com/libvips/libvips.git
cd libvips
./autogen.sh
make -j
sudo make install
vips --vips-version
popd

