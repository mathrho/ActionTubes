
cmake_minimum_required(VERSION 2.4)

set(SRCS
  compute_optical_flow.cpp
  )

# OpenCV Config
find_package(OpenCV REQUIRED)
message("-- OPENCV include:   " ${OpenCV_INCLUDE_DIRS})
message("-- OPENCV libs dir:  " ${OpenCV_LIB_DIR})
message("-- OPENCV libs:   " ${OpenCV_LIBS} )

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIB_DIR})

ADD_DEFINITIONS(-DDEBUG -ggdb -O3 -Wall -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -fbranch-probabilities)

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

################################

ADD_EXECUTABLE(
  compute_optical_flow
  compute_optical_flow.cpp
  ${SRCS}
)

TARGET_LINK_LIBRARIES(
  compute_optical_flow
  ${OPENCV_LIBRARIES}
  ${OpenCV_LIBS}
)

#ADD_EXECUTABLE(
#  farneback_flow
#  farneback_flow.cpp
#  ${SRCS}
#)

#TARGET_LINK_LIBRARIES(
#  farneback_flow
#  ${OPENCV_LIBRARIES}
#  ${OpenCV_LIBS}
#)

#ADD_EXECUTABLE(
# # lk_flow
#  lk_flow.cpp
#  ${SRCS}
#)

#TARGET_LINK_LIBRARIES(
#  lk_flow
#  ${OPENCV_LIBRARIES}
#  ${OpenCV_LIBS}
#)

################################

foreach(dir ${DIRS})
  FILE(GLOB files ${dir}/*.h*)
  LIST(APPEND h_files ${files})
endforeach(dir)
add_custom_target(TAGS etags --members -l c++ ${h_files} ${SRCS})

