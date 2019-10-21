#!/bin/bash

if [[ -d "/opt/opencv4" ]]; then
  echo Already installed!
  exit 0
fi

echo Downloading OpenCV4...
curl -sL https://storage.googleapis.com/opencv4/opencv4.tgz | tar zxf - -C / \
  && ldconfig /opt/opencv4/lib/ \
  && rm -f /usr/lib/pkgconfig/opencv4.pc \
  && ln -s /opt/opencv4/lib/pkgconfig/opencv4.pc /usr/lib/pkgconfig/opencv4.pc

