#!/bin/sh

set -e

CXX_FLAGS="-I$PWD/../libmnn/include -std=c++11 -ffunction-sections -fdata-sections -Ofast -fPIC"
LD_FLAGS="-L$PWD/../libmnn/lib -lMNN -lpthread -Wl,-gc-sections -Wl,-strip-all -flto"

case "$1" in
"")
    $CXX -Wall -D_TEST_ bmpfile.c facedet.cpp $CXX_FLAGS $LD_FLAGS -o test
    case "$TARGET_PLATFORM" in
    win32)
        $CXX --shared facedet.cpp $CXX_FLAGS $LD_FLAGS -o facedet.dll
        dlltool -l facedet.lib -d facedet.def
        $STRIP *.exe *.dll
        ;;
    ubuntu)
        $CXX --shared facedet.cpp $CXX_FLAGS $LD_FLAGS -o libfacedet.so
        $STRIP test *.so
        ;;
    msc33x)
        $CXX --shared facedet.cpp $CXX_FLAGS $LD_FLAGS -o libfacedet.so
        $STRIP test *.so
        $CXX -Wall -c bmpfile.c facedet.cpp $CXX_FLAGS $LD_FLAGS
        cp $PWD/../libmnn/lib/libMNN.a libfacedet.a
        ${CC}-ar rcs libfacedet.a bmpfile.o facedet.o
        ;;
    esac
    ;;
clean)
    rm -rf test *.so *.dll *.exe out.bmp
    ;;
esac
