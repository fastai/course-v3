#!/usr/bin/env bash
set -xe

cd fastai-video-browser
npm run build
cp -r build/* ../docs/videos/
rm -rf ../docs/static
mv ../docs/videos/static ../docs/
