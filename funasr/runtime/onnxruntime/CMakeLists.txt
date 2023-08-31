cmake_minimum_required(VERSION 3.16)

project(FunASROnnx)

option(ENABLE_GLOG "Whether to build glog" ON)
option(GPU "Whether to build with GPU" OFF)

# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


include(TestBigEndian)
test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
    message("Big endian system")
else()
    message("Little endian system")
endif()

# for onnxruntime
IF(WIN32)
	if(CMAKE_CL_64)
		link_directories(${ONNXRUNTIME_DIR}\\lib)
	else()
		add_definitions(-D_WIN_X86)
	endif()
ELSE()
    link_directories(${ONNXRUNTIME_DIR}/lib)
    link_directories(${FFMPEG_DIR}/lib)
endif()

#libtorch
if(GPU)
    add_definitions(-DUSE_GPU)
    set(TORCH_DIR "/opt/conda/lib/python3.7/site-packages/torch")
    include_directories(${TORCH_DIR}/include)
    include_directories(${TORCH_DIR}/include/torch/csrc/api/include)
    link_directories(${TORCH_DIR}/lib)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

include_directories(${PROJECT_SOURCE_DIR}/third_party/kaldi-native-fbank)
include_directories(${PROJECT_SOURCE_DIR}/third_party/yaml-cpp/include)

if(ENABLE_GLOG)
    include_directories(${PROJECT_SOURCE_DIR}/third_party/glog)
    set(BUILD_TESTING OFF)
    set(WITH_GFLAGS OFF)
    add_subdirectory(third_party/glog)
endif()

add_subdirectory(third_party/yaml-cpp)
add_subdirectory(third_party/kaldi-native-fbank/kaldi-native-fbank/csrc)
add_subdirectory(src)
add_subdirectory(bin)
