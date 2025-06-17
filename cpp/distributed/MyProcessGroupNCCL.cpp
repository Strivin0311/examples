#ifdef USE_MY_C10D_NCCL

#include "MyProcessGroupNCCL.hpp"

namespace myc10d {
    MyProcessGroupNCCL::MyProcessGroupNCCL(
        c10::intrusive_ptr<c10d::Store> store,
        int rank,
        int size,
        c10::intrusive_ptr<Options> options) : ProcessGroupNCCL(store, rank, size, options) {}

    MyProcessGroupNCCL::~MyProcessGroupNCCL() = default;

    // get all nccl streams
    std::unordered_map<std::string, at::cuda::CUDAStream>& MyProcessGroupNCCL::getNCCLStreams() {
        return ncclStreams_; // this is a protected member in the parent class
    }

    // get the nccl stream w.r.t. the device
    at::cuda::CUDAStream& MyProcessGroupNCCL::getNCCLStream(int device) {
        device = device == -1 ? getDevice() : device;
        std::string device_str = std::to_string(device);

        TORCH_CHECK(
            ncclStreams_.find(device_str) != ncclStreams_.end(), 
            "NCCL stream for device " + device_str + " not found"
        );
        
        return ncclStreams_.at(device_str);
    }
}

#endif // USE_MY_C10D_NCCL