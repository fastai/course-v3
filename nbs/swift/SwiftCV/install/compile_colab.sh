#!/bin/bash

# Execute this script from Colab notebook, e.g.:
# !git clone https://github.com/fastai/course-v3
# !course-v3/nbs/swift/SwiftCV/install/compile_colab.sh
# Resulting `opencv4.tgz` file will appear in `course-v3/nbs/swift/SwiftCV/install` folder

# Download file using Colab's Files panel or move to storage of your choice, e.g. GCP:
# from google.colab import auth
# auth.authenticate_user()
# !gcloud config set project your_project
# !gsutil cp course-v3/nbs/swift/SwiftCV/install/opencv4.tgz gs://your_bucket/
# !gsutil acl set public-read gs://{bucket_name}/opencv4.tgz

INSTALL_DIR=/opt/opencv4/
SCRIPT_DIR=$(dirname $(readlink -f $0))

mkdir -p $INSTALL_DIR
pushd $SCRIPT_DIR
INSTALL_PREFIX=$INSTALL_DIR ./install_cv4.sh
popd

tar vczf $SCRIPT_DIR/opencv4.tgz $INSTALL_DIR -C /
