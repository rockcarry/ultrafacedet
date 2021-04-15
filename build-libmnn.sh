#!/bin/sh

set -e

TOPDIR=$PWD

if [ ! -d MNN ]; then
#   git clone https://github.com/alibaba/MNN.git
    git clone https://github.com.cnpmjs.org/alibaba/MNN.git
fi

cd $TOPDIR/MNN
git checkout 1.1.5
git checkout .
cd -

rm -rf $TOPDIR/build-mnn
mkdir -p $TOPDIR/build-mnn
cd $TOPDIR/build-mnn

cmake $TOPDIR/MNN \
-DCMAKE_INSTALL_PREFIX=$TOPDIR/build-mnn/install \
-DMNN_BUILD_SHARED_LIBS=OFF \
-DMNN_OPENMP=OFF \
-DMNN_BUILD_TOOLS=OFF \
-DMNN_BUILD_QUANTOOLS=OFF \
-DMNN_FORBID_MULTI_THREAD=ON \
-DMNN_USE_THREAD_POOL=OFF \
-DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE
make -j8 && make install

rm -rf $TOPDIR/libmnn
mv $TOPDIR/build-mnn/install/ $TOPDIR/libmnn
rm -rf $TOPDIR/build-mnn

cd $TOPDIR
