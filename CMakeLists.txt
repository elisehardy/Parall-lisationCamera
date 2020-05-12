CMAKE_MINIMUM_REQUIRED(VERSION 3.7)

PROJECT(Parallelization)

OPTION(PROFILING "Profile the time taken by each step" ON)


############################## 3RD PARTIES LIBS ################################

SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fopenmp)

FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(
    SYSTEM
    ${OpenCV_DIR}
)

LINK_LIBRARIES(
    ${OpenCV_LIBS}
)

################################# Project ##################################

SET(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/)

FILE(GLOB_RECURSE SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
FILE(GLOB_RECURSE HEADER_FILES ${CMAKE_SOURCE_DIR}/include/*.hpp)

INCLUDE_DIRECTORIES(
    ${INCLUDE_DIR}
)

SET(TARGET_NAME cameraParallelization)
ADD_EXECUTABLE(${TARGET_NAME} ${SOURCE_FILES} ${HEADER_FILES})

IF (PROFILING)
    TARGET_COMPILE_DEFINITIONS(${TARGET_NAME} PUBLIC PROFILING)
ENDIF ()

TARGET_COMPILE_OPTIONS(
    ${TARGET_NAME} PRIVATE
    -O3
    -std=c++17
    -Wno-c++98-compat
    -Wno-gnu-statement-expression
)
# Adding as much warning as possible on GNU gcc/g++ and Clang
IF (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    TARGET_COMPILE_OPTIONS(
        ${TARGET_NAME} PRIVATE
        -Wall
        -Wextra
        -Werror
    )
ELSEIF (CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang")
    TARGET_COMPILE_OPTIONS(
        ${TARGET_NAME} PRIVATE
        -Weverything
        -Wfatal-errors
    )
ENDIF()
