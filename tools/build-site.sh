#!/usr/bin/env bash
set -xe

cd fastai-video-browser
npm run build
rm -rf ../docs/videos/*
cp -r build/* ../docs/videos/
rm -rf ../docs/static
mv ../docs/videos/static ../docs/
