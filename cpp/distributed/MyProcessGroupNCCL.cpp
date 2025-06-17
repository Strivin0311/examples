#ifdef USE_MY_C10D_NCCL

#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
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
        std::string deviceStr = std::to_string(device);

        TORCH_CHECK(
            ncclStreams_.find(deviceStr) != ncclStreams_.end(), 
            "NCCL stream for device " + deviceStr + " not found"
        );
        
        return ncclStreams_.at(deviceStr);
    }

    // get all nccl comms
    std::unordered_map<std::string, std::shared_ptr<c10d::NCCLComm>>& MyProcessGroupNCCL::getTorchNCCLComms() {
        return devNCCLCommMap_; // this is a protected member in the parent class
    }

    // get the nccl comm w.r.t. the device
    std::shared_ptr<c10d::NCCLComm> MyProcessGroupNCCL::getTorchNCCLComm(int device) {
        device = device == -1 ? getDevice() : device;
        std::string deviceStr = std::to_string(device);

        TORCH_CHECK(
            ncclStreams_.find(deviceStr) != ncclStreams_.end(), 
            "NCCL stream for device " + deviceStr + " not found"
        );
        
        return devNCCLCommMap_.at(deviceStr);
    }

    /** NOTE: only provided in the main branch >= v-2.7.1 */
    // get the nccl comm ptr w.r.t the current device
    // int64_t MyProcessGroupNCCL::getNCCLCommPtr() {
    //     return c10d::ProcessGroupNCCL::getCommPtr();
    // }
    
} // namespace myc10d

#endif // USE_MY_C10D_NCCL