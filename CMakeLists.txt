cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(test)

find_package(Boost REQUIRED)
set(OPENCV_MANUAL_PATH /usr/local/include/opencv /usr/local/include/opencv2)

IF (APPLE)
set (GLOG_PATH /usr/local/Cellar/glog/0.3.3/include/glog)
ELSE (APPLE)
set (GLOG_PATH /usr/local/include/glog)
set (BOOST_PATH /usr/local/include/boost)
ENDIF (APPLE)
set(RESEARCH_ROOT_PATH ${PROJECT_SOURCE_DIR}/../)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${OPENCV_MANUAL_PATH})

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${GLOG_PATH})

include_directories(${RESEARCH_ROOT_PATH})

# for static
set(GLOG_LIBS glog)

set(MANUAL_CV_LIBS 
  opencv_core 
  opencv_imgproc
  opencv_video
  opencv_highgui
  opencv_objdetect 
  opencv_nonfree 
  opencv_flann
  opencv_ml
  opencv_features2d)

IF (APPLE) 
set(BOOST_LIBS 
  boost_program_options-mt 
  boost_system-mt 
  boost_filesystem-mt
  boost_date_time-mt)
ELSE (APPLE)
set(BOOST_LIBS 
  boost_program_options 
  boost_system 
  boost_filesystem
  boost_date_time)
ENDIF (APPLE)

# dynamic
link_directories(/usr/local/lib)

add_executable(test
  test.cpp 
  fuzzy_clustering.cpp)
 
target_link_libraries(
  test
  ${MANUAL_CV_LIBS} 
  ${GLOG_LIBS} 
  ${BOOST_LIBS}
  )

