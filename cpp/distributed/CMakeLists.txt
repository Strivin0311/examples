cmake_minimum_required(VERSION 3.5)
project(dist-mnist)

find_package(Torch REQUIRED)

find_package(MPI REQUIRED)

option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../tools/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

add_definitions(-DUSE_C10D_MPI)
add_definitions(-DUSE_C10D_NCCL)
add_definitions(-DUSE_MY_C10D_NCCL)

include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

add_executable(dist-mnist dist-mnist.cpp MyProcessGroupNCCL.cpp)
target_link_libraries(dist-mnist ${TORCH_LIBRARIES})
target_link_libraries(dist-mnist ${MPI_LIBRARIES})
target_link_libraries(dist-mnist ${CMAKE_PREFIX_PATH}/lib/libc10.so)
target_link_libraries(dist-mnist ${CMAKE_PREFIX_PATH}/lib/libc10_cuda.so)

if(MPI_COMPILE_FLAGS)
  set_target_properties(dist-mnist PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(dist-mnist PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
