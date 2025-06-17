#ifdef USE_MY_C10D_NCCL

#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/torch.h>


namespace myc10d {
    class TORCH_API MyProcessGroupNCCL : public c10d::ProcessGroupNCCL {
    public:
        MyProcessGroupNCCL(
            c10::intrusive_ptr<c10d::Store> store,
            int rank,
            int size,
            c10::intrusive_ptr<Options> options = Options::create());
    
        ~MyProcessGroupNCCL() override;

        // get all nccl streams
        std::unordered_map<std::string, at::cuda::CUDAStream>& getNCCLStreams();

        // get the nccl stream w.r.t. the device
        at::cuda::CUDAStream& getNCCLStream(int device = -1);

        // get all torch nccl comms
        std::unordered_map<std::string, std::shared_ptr<c10d::NCCLComm>>& getTorchNCCLComms();

        // get the torch nccl comm w.r.t. the device
        std::shared_ptr<c10d::NCCLComm> getTorchNCCLComm(int device = -1);

        /** NOTE: only provided in the main branch >= v-2.7.1 */
        // get the nccl comm ptr w.r.t the current device
        // int64_t getNCCLCommPtr();

        // set the device
        /** NOTE: another (maybe better) way is to just use the current device
         * auto device = at::Device(at::kCUDA, at::cuda::current_device());
         * std::string deviceKey = std::to_string(device.index());
         */
        void setDevice(int device) { device_ = device; }

        int getDevice() const { return device_; }

        int device_ = -1;
    };
} // namespace myc10d

#endif // USE_MY_C10D_NCCL