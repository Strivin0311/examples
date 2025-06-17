#ifdef USE_MY_C10D_NCCL

#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
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

        // set the device
        void setDevice(int device) { device_ = device; }

        int getDevice() const { return device_; }

        int device_ = -1;
    };
} // namespace myc10d

#endif // USE_MY_C10D_NCCL