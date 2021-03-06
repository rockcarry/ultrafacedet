set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR armv7-a)

set(CMAKE_C_COMPILER "arm-buildroot-linux-uclibcgnueabihf-gcc")
set(CMAKE_CXX_COMPILER "arm-buildroot-linux-uclibcgnueabihf-g++")

set(MNN_FORBID_MULTI_THREAD ON)
set(MNN_USE_THREAD_POOL OFF)

