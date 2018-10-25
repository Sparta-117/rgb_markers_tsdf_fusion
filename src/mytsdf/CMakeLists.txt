cmake_minimum_required(VERSION 2.8.3)
project(mytsdf)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++0x")

############### dependencies ######################
# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
#g2o
find_package( G2O REQUIRED )
include_directories(${G2O_INCLUDE_DIRS})
LIST(APPEND G2O_LIBS
  cxsparse
  cholmod
  g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
  )
#PCL
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
#gpu
option(USE_CUDA "Use CUDA" ON)
find_package(CUDA REQUIRED)
find_package(CUDA 8.0)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--disable-warnings;--ptxas-options=-v;-use_fast_math;-lineinfo;-std=c++11)

#CSparse
find_package( CSparse REQUIRED )
include_directories(${CSPARSE_INCLUDE_DIR})
#Opencv
find_package(OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIR})
# Eigen
include_directories( "/usr/include/eigen3" )
include_directories(${EIGEN_INCLUDE_DIRS})
# Sophus 
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )
#ros
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
  rgbd_srv
)
catkin_package()
include_directories(${catkin_INCLUDE_DIRS})
#Pangolin
find_package(Pangolin 0.4 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

#self file
# include_directories(../../devel/include)
include_directories(include/mytsdf)
include_directories(include)
include_directories( ${PROJECT_SOURCE_DIR}/include )

### solve the conflict between PCL and cuda
get_directory_property(dir_defs DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
set(vtk_flags)
foreach(it ${dir_defs})
    if(it MATCHES "vtk*")
    list(APPEND vtk_flags ${it})
    endif()
endforeach()

foreach(d ${vtk_flags})
    remove_definitions(-D${d})
endforeach()
###

############### execute ######################
file(GLOB cpu_source_files "${PROJECT_SOURCE_DIR}/src/*.cpp")
#file(GLOB gpu_source_files "${PROJECT_SOURCE_DIR}/src/*.cu")
set(SRC src/camera.cpp src/config.cpp src/qrplane.cpp)

#add_executable(mytsdf src/mytsdf.cpp ${SRC})
#add_executable(mytsdf src/mytsdf.cpp ${cpu_source_files})

CUDA_ADD_EXECUTABLE(mytsdf src/mytsdf.cpp ${cpu_source_files} src/tsdf_gpu.cu)
#CUDA_ADD_LIBRARY(cuda_gpu src/gpu.cu)

target_link_libraries(mytsdf
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
   ${PCL_LIBRARIES}
   ${Sophus_LIBRARIES}
   ${CSPARSE_LIBRARY}
   ${G2O_LIBS}
   ${CUDA_LIBRARIES}
   ${Eigen_LIBRARIES}
   ${Pangolin_LIBRARIES}
   #cuda_gpu
 )

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
