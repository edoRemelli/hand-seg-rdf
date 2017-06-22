# To find OpenCV 2 library visit http://opencv.willowgarage.com/wiki/
#
# The follwoing variables are optionally searched for defaults
#  OpenCV2_ROOT_DIR:                   Base directory of OpenCV 2 tree to use.
#
# The following are set after configuration is done:
#  OpenCV_FOUND
#  OpenCV_INCLUDE_DIRS
#  OpenCV_LIBRARIES
#
# $Id: $
#  
# Balazs [2011-01-18]:
# - Created from scratch for the reorganized OpenCV 2 structure introduced at version 2.2
# Jbohren [2011-06-10]:
# - Added OpenCV_ROOT_DIR for UNIX platforms & additional opencv include dir
# jmorrison [2013-11-14]:
# - Added flag to disable GPU requirement (NO_OPENCV_GPU)
# 
# This file should be removed when CMake will provide an equivalent

#--- Select exactly ONE OpenCV 2 base directory to avoid mixing different version headers and libs
find_path(OpenCV_ROOT_INC_DIR NAMES opencv2/opencv.hpp
    PATHS
        C:/Developer/OpenCV-2.4.10/build/include                # Windows
        "$ENV{OpenCV_ROOT_DIR}/include"     # *NIX: custom install
        /usr/local/include                  # Linux: default dir by CMake
        /usr/include                        # Linux
        /opt/local/include                  # OS X: default MacPorts location
        NO_DEFAULT_PATH)


find_path(OpenCV_CORE_INCLUDE_DIR       NAMES core.hpp         PATHS "${OpenCV_ROOT_INC_DIR}/opencv2/core")
find_path(OpenCV_IMGPROC_INCLUDE_DIR    NAMES imgproc.hpp      PATHS "${OpenCV_ROOT_INC_DIR}/opencv2/imgproc")
find_path(OpenCV_CONTRIB_INCLUDE_DIR    NAMES contrib.hpp      PATHS "${OpenCV_ROOT_INC_DIR}/opencv2/contrib")
find_path(OpenCV_HIGHGUI_INCLUDE_DIR    NAMES highgui.hpp      PATHS "${OpenCV_ROOT_INC_DIR}/opencv2/highgui")
find_path(OpenCV_FLANN_INCLUDE_DIR      NAMES flann.hpp        PATHS "${OpenCV_ROOT_INC_DIR}/opencv2/flann")

set(OpenCV_INCLUDE_DIRS
    ${OpenCV_ROOT_INC_DIR}
    ${OpenCV_ROOT_INC_DIR}/opencv2
    ${OpenCV_CORE_INCLUDE_DIR}
    ${OpenCV_IMGPROC_INCLUDE_DIR}
    ${OpenCV_CONTRIB_INCLUDE_DIR}
    ${OpenCV_HIGHGUI_INCLUDE_DIR}
    ${OpenCV_FLANN_INCLUDE_DIR})

#--- Specify where DLL is searched for
list(APPEND OPENCV_LIBRARY_SEARCH_PATHS $ENV{OpenCV_ROOT_DIR})
list(APPEND OPENCV_LIBRARY_SEARCH_PATHS "C:/Developer/OpenCV-2.4.10/build/x64/vc12/lib")
list(APPEND OPENCV_LIBRARY_SEARCH_PATHS "/usr/local/lib")
list(APPEND OPENCV_LIBRARY_SEARCH_PATHS "/opt/local/lib")
list(APPEND OPENCV_LIBRARY_SEARCH_PATHS "/usr/lib")


#--- FIND RELEASE LIBRARIES
find_library(OpenCV_CORE_LIBRARY_REL       NAMES opencv_core opencv_core230 opencv_core220 opencv_core2410                         PATHS ${OPENCV_LIBRARY_SEARCH_PATHS})
find_library(OpenCV_IMGPROC_LIBRARY_REL    NAMES opencv_imgproc opencv_imgproc230 opencv_imgproc220 opencv_imgproc2410             PATHS ${OPENCV_LIBRARY_SEARCH_PATHS})
find_library(OpenCV_CONTRIB_LIBRARY_REL    NAMES opencv_contrib opencv_contrib230 opencv_contrib220 opencv_contrib2410             PATHS ${OPENCV_LIBRARY_SEARCH_PATHS})
find_library(OpenCV_HIGHGUI_LIBRARY_REL    NAMES opencv_highgui opencv_highgui230 opencv_highgui220 opencv_highgui2410             PATHS ${OPENCV_LIBRARY_SEARCH_PATHS})
list(APPEND OpenCV_LIBRARIES_REL ${OpenCV_CORE_LIBRARY_REL})
list(APPEND OpenCV_LIBRARIES_REL ${OpenCV_IMGPROC_LIBRARY_REL})
list(APPEND OpenCV_LIBRARIES_REL ${OpenCV_CONTRIB_LIBRARY_REL})
list(APPEND OpenCV_LIBRARIES_REL ${OpenCV_HIGHGUI_LIBRARY_REL})

#--- FIND DEBUG LIBRARIES
if(WIN32)
    find_library(OpenCV_CORE_LIBRARY_DEB       NAMES opencv_cored opencv_core230d opencv_core220d opencv_core2410d                     PATHS ${OPENCV_LIBRARY_SEARCH_PATHS})
    find_library(OpenCV_IMGPROC_LIBRARY_DEB    NAMES opencv_imgprocd opencv_imgproc230d opencv_imgproc220d opencv_imgproc2410d         PATHS ${OPENCV_LIBRARY_SEARCH_PATHS})
    find_library(OpenCV_CONTRIB_LIBRARY_DEB    NAMES opencv_contribd opencv_contrib230d opencv_contrib220d opencv_contrib2410d         PATHS ${OPENCV_LIBRARY_SEARCH_PATHS})
    find_library(OpenCV_HIGHGUI_LIBRARY_DEB    NAMES opencv_highguid opencv_highgui230d opencv_highgui220d opencv_highgui2410d         PATHS ${OPENCV_LIBRARY_SEARCH_PATHS})
    list(APPEND OpenCV_LIBRARIES_DEB ${OpenCV_CORE_LIBRARY_DEB})
    list(APPEND OpenCV_LIBRARIES_DEB ${OpenCV_IMGPROC_LIBRARY_DEB})
    list(APPEND OpenCV_LIBRARIES_DEB ${OpenCV_CONTRIB_LIBRARY_DEB})
    list(APPEND OpenCV_LIBRARIES_DEB ${OpenCV_HIGHGUI_LIBRARY_DEB})
endif()

#--- Setup cross-config libraries
set(OpenCV_LIBRARIES "")
if(WIN32)
    list(APPEND OpenCV_LIBRARIES optimized ${OpenCV_CORE_LIBRARY_REL}    debug ${OpenCV_CORE_LIBRARY_DEB})
    list(APPEND OpenCV_LIBRARIES optimized ${OpenCV_IMGPROC_LIBRARY_REL} debug ${OpenCV_IMGPROC_LIBRARY_DEB})
    list(APPEND OpenCV_LIBRARIES optimized ${OpenCV_CONTRIB_LIBRARY_REL} debug ${OpenCV_CONTRIB_LIBRARY_DEB})
    list(APPEND OpenCV_LIBRARIES optimized ${OpenCV_HIGHGUI_LIBRARY_REL} debug ${OpenCV_HIGHGUI_LIBRARY_DEB})
else()
    list(APPEND OpenCV_LIBRARIES ${OpenCV_CORE_LIBRARY_REL}   )
    list(APPEND OpenCV_LIBRARIES ${OpenCV_IMGPROC_LIBRARY_REL})
    list(APPEND OpenCV_LIBRARIES ${OpenCV_CONTRIB_LIBRARY_REL})
    list(APPEND OpenCV_LIBRARIES ${OpenCV_HIGHGUI_LIBRARY_REL})
endif()

#--- Verifies everything (include) was found
set(OpenCV_FOUND ON)
FOREACH(NAME ${OpenCV_INCLUDE_DIRS})
    IF(NOT EXISTS ${NAME})
        message(WARNING "Could not find: ${NAME}")
        set(OpenCV_FOUND OFF)
    endif(NOT EXISTS ${NAME})
ENDFOREACH(NAME)

#--- Verifies everything (release lib) was found
FOREACH(NAME ${OpenCV_LIBRARIES_REL})
    IF(NOT EXISTS ${NAME})
        message(WARNING "Could not find: ${NAME}")
        set(OpenCV_FOUND OFF)
    endif(NOT EXISTS ${NAME})
 ENDFOREACH()

#--- Verifies everything (debug lib) was found
FOREACH(NAME ${OpenCV_LIBRARIES_DEB})
    IF(NOT EXISTS ${NAME})
        message(WARNING "Could not find: ${NAME}")
        set(OpenCV_FOUND OFF)
    endif(NOT EXISTS ${NAME})
ENDFOREACH()

#--- Display help message
IF(OpenCV_FOUND)
	MESSAGE(STATUS "OpenCV 2 IS FOUND!")
	 set(OpenCV_FOUND ON)
endif()
IF(NOT OpenCV_FOUND)
    IF(OpenCV_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "OpenCV 2 not found.")
    else()
        MESSAGE(STATUS "OpenCV 2 not found.")
    endif()
endif()
