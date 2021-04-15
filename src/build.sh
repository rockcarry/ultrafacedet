#!/bin/sh

set -e

CXX_FLAGS="-I$PWD/../libmnn/include -std=c++11 -ffunction-sections -fdata-sections -Os -fPIC"
LD_FLAGS="-L$PWD/../libmnn/lib -lMNN -lpthread -Wl,-gc-sections -Wl,-strip-all -flto"

case "$1" in
"")
    $CXX -Wall -D_TEST_ bmpfile.c ultrafacedet.cpp $CXX_FLAGS $LD_FLAGS -o test
    case "$TARGET_PLATFORM" in
    win32)
        $CXX --shared ultrafacedet.cpp $CXX_FLAGS $LD_FLAGS -o ultrafacedet.dll
        dlltool -l ultrafacedet.lib -d ultrafacedet.def
        $STRIP *.exe *.dll
        ;;
    ubuntu)
        $CXX --shared ultrafacedet.cpp $CXX_FLAGS $LD_FLAGS -o ultrafacedet.so
        $STRIP test *.so
        ;;
    msc33x)
        $CXX --shared ultrafacedet.cpp $CXX_FLAGS $LD_FLAGS -o ultrafacedet.so
        $STRIP test *.so
        ;;
    esac
    ;;
clean)
    rm -rf test *.so *.dll *.exe out.bmp
    ;;
esac
