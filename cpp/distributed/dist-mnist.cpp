/** 
 * NOTE: we've modified the source code according to this pr: https://github.com/pytorch/examples/pull/1341
*/
#ifndef USE_C10D_MPI
#define USE_C10D_MPI
#endif

#ifndef USE_C10D_NCCL
#define USE_C10D_NCCL
#endif

#ifndef USE_MY_C10D_NCCL
#define USE_MY_C10D_NCCL
#endif

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/torch.h>
#include <c10/util/env.h>

#include <cuda_profiler_api.h>
#include "nvtx3/nvToolsExt.h"

#include "mpi.h"
#include "nccl.h"

#include "MyProcessGroupNCCL.hpp"


#define USE_NCCL_AS_COMM_BACKEND
#define USE_MY_NCCL_AS_COMM_BACKEND


#define MPI_CHECK(cmd)                                                   \
  do {                                                                   \
    int mpiStatus = cmd;                                                 \
    if (mpiStatus != MPI_SUCCESS) {                                      \
      std::string err = "MPI error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) +                                     \
          ", with error code: " + std::to_string(mpiStatus);             \
      TORCH_CHECK(false, err);                                           \
    }                                                                    \
  } while (0)



class Logger {
public:
    Logger(std::string prefix, int rank, bool append = false) {
        std::string filename = prefix + "_r" + std::to_string(rank) + ".log";
        logFile.open(filename, std::ios_base::out | (append ? std::ios_base::app : std::ios_base::trunc));
        if (!logFile.is_open()) {
            std::cerr << "Failed to open log file: " << filename << std::endl;
        }
    }

    void log(const std::string& message) {
        if (logFile.is_open()) {
            logFile << message << std::endl;
        }
    }

    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }

private:
    std::ofstream logFile;
};


// Define a Convolutional Module
struct ModelImpl : torch::nn::Module {
  ModelImpl()
      : conv1(torch::nn::Conv2dOptions(1, 10, 5)),
        conv2(torch::nn::Conv2dOptions(10, 20, 5)),
        fc1(320, 50),
        fc2(50, 10) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv2_drop", conv2_drop);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  at::Tensor forward(at::Tensor x) {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::relu(
        torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, 0.5, is_training());
    x = fc2->forward(x);
    return torch::log_softmax(x, 1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::Dropout2d conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

// use TORCH_MODULE macor wrapper to avoid explicit shared ptr creation
// see: https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html
TORCH_MODULE(Model);


template <typename ProcessGroupBackend>
void waitWork(
    c10::intrusive_ptr<ProcessGroupBackend> pg,
    std::vector<c10::intrusive_ptr<c10d::Work>> works) {
  for (auto& work : works) {
    try {
      work->wait();
    } catch (const std::exception& ex) {
      std::cerr << "ProcessGroup Exception received: " << ex.what() << std::endl;
      pg->abort();
    }
  }
}


std::string parseLauncher(int argc, char* argv[]) {
  std::string launcher = "mpirun"; int opt;
  while ((opt = getopt(argc, argv, "l:")) != -1) {
      switch (opt) {
          case 'l':
              launcher = optarg;
              break;
          default: /* '?' */
              std::cerr << "Usage: " << argv[0] << " -l launcher {mpirun|torchrun}" << std::endl;
              exit(EXIT_FAILURE);
      }
  }

  // reset opt to allow reparsing in the future
  optind = 1; // reset optind to 1
  optarg = nullptr; // reset optarg
  
  return launcher;
}


c10::intrusive_ptr<c10d::ProcessGroupNCCL> createProcessGroupNCCL(int argc, char* argv[]) {
  std::string launcher = parseLauncher(argc, argv);

  int rank, num_ranks;
  if (launcher == "mpirun") {
    // Check if MPI was already initialized
    int mpi_was_initialized = 0;
    MPI_CHECK(MPI_Initialized(&mpi_was_initialized));
    std::cout << "Was MPI initialized: " << mpi_was_initialized << std::endl;
    if (mpi_was_initialized != 0) {
      /** TODO: do something here if MPI was already initialized */
    }
    
    // Initialize MPI explicitly
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_ranks));
    std::cout << "[Rank " << rank << "] " << "MPI initialized for NCCL" << std::endl;
  }
  else if (launcher == "torchrun") {
    rank = std::stoi(getenv("RANK"));
    num_ranks = std::stoi(getenv("WORLD_SIZE"));
    std::cout << "[Rank " << rank << "] " << "TorchRun initialized for NCCL" << std::endl;
  }
  else {
    std::stringstream ss; ss << "Unsupported launcher: " << launcher;
    TORCH_CHECK(false, ss.str());
  }
  
  // Init store
  std::string master_addr(getenv("MASTER_ADDR"));
  uint16_t master_port = std::stoi(getenv("MASTER_PORT"));
  if (launcher == "torchrun") {
    /** NOTE: when using torchrun, the master port is already occupied
     * thus here we increment the master port to use
     */
    master_port++;
  }
  c10d::TCPStoreOptions store_options = {
    .port=master_port,
    /** NOTE: this is necessary to determine the master rank,
     * otherwise the initialization will hang
     */
    .isServer=rank == 0,
  };
  auto store = c10::make_intrusive<c10d::TCPStore>(master_addr, store_options);
  /** NOTE: PrefixStore is no use to the address-occupied problem
   * I guess we need to pass in the exact store object that torchrun created
   */
  // store = c10::make_intrusive<c10d::PrefixStore>("dist-mnist/", store);

  // Init nccl process group ptr
  return c10::make_intrusive<c10d::ProcessGroupNCCL>(store, rank, num_ranks);
}


c10::intrusive_ptr<myc10d::MyProcessGroupNCCL> createMyProcessGroupNCCL(int argc, char* argv[]) {
  std::string launcher = parseLauncher(argc, argv);

  int rank, num_ranks;
  if (launcher == "mpirun") {
    // Check if MPI was already initialized
    int mpi_was_initialized = 0;
    MPI_CHECK(MPI_Initialized(&mpi_was_initialized));
    std::cout << "Was MPI initialized: " << mpi_was_initialized << std::endl;
    if (mpi_was_initialized != 0) {
      /** TODO: do something here if MPI was already initialized */
    }
    
    // Initialize MPI explicitly
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_ranks));
    std::cout << "[Rank " << rank << "] " << "MPI initialized for MyNCCL" << std::endl;
  }
  else if (launcher == "torchrun") {
    rank = std::stoi(getenv("RANK"));
    num_ranks = std::stoi(getenv("WORLD_SIZE"));
    std::cout << "[Rank " << rank << "] " << "TorchRun initialized for MyNCCL" << std::endl;
  }
  else {
    std::stringstream ss; ss << "Unsupported launcher: " << launcher;
    TORCH_CHECK(false, ss.str());
  }
  
  // Init store
  std::string master_addr(getenv("MASTER_ADDR"));
  uint16_t master_port = std::stoi(getenv("MASTER_PORT"));
  if (launcher == "torchrun") {
    /** NOTE: when using torchrun, the master port is already occupied
     * thus here we increment the master port to use
     */
    master_port++;
  }
  c10d::TCPStoreOptions store_options = {
    .port=master_port,
    /** NOTE: this is necessary to determine the master rank,
     * otherwise the initialization will hang
     */
    .isServer=rank == 0,
  };
  auto store = c10::make_intrusive<c10d::TCPStore>(master_addr, store_options);
  /** NOTE: PrefixStore is no use to the address-occupied problem
   * I guess we need to pass in the exact store object that torchrun created
   */
  // store = c10::make_intrusive<c10d::PrefixStore>("dist-mnist/", store);

  // Init my nccl process group ptr
  auto pg = c10::make_intrusive<myc10d::MyProcessGroupNCCL>(store, rank, num_ranks);
  
  // Set device
  pg->setDevice(rank);

  return pg;
}


void printMainArgs(int argc, char* argv[]) {
  std::cout << "Printing main arguments: " << std::endl;
  std::cout << "Number of arguments: " << argc << std::endl;
  std::cout << "Arguments:" << std::endl;

  for (int i = 0; i < argc; ++i) {
      std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
  } std::cout << std::endl;
}


void printDistEnvVars() {
  std::cout << "Printing dist-environment variables: " << std::endl;
  std::cout << "MASTER_ADDR: " << getenv("MASTER_ADDR") << std::endl;
  std::cout << "MASTER_PORT: " << getenv("MASTER_PORT") << std::endl;
  std::cout << std::endl;
}


void printProcessGroupBackend() {
  std::cout << "Printing process group backend: ";
  #ifdef USE_NCCL_AS_COMM_BACKEND
    #ifdef USE_MY_NCCL_AS_COMM_BACKEND
      std::cout << "MyNCCL" << std::endl;
    #else
      std::cout << "NCCL" << std::endl;
    #endif
  #else
  std::cout << "MPI" << std::endl;
  #endif
  std::cout << std::endl;
}


std::string getNCCLUIDString(ncclUniqueId& uid, int width = 4) {
  std::stringstream ss;
  for (int i = 0; i < width; ++i) {
      ss << std::hex << std::setw(2) << std::setfill('0') << (int)(unsigned char)uid.internal[i];
  }
  return ss.str();
}


void analysisMyNCCL(Logger& logger, c10::intrusive_ptr<myc10d::MyProcessGroupNCCL> pg, int num_ranks, int rank) {
  std::string sep(50, '='); std::string subSep(35, '-');
  std::stringstream ss;

  ss << sep << " Analysis MyNCCL " << sep << std::endl;
  
  ss << subSep << " All NCCL streams " << subSep << std::endl;
  auto streams = pg->getNCCLStreams();
  for (const auto& pair : streams) {
    ss << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
  }

  int device = pg->getDevice(); std::string deviceStr = std::to_string(device);
  ss << subSep << " NCCL stream for current device " << device << " " << subSep << std::endl;
  ss << pg->getNCCLStream() << std::endl;

  // ncclUniqueId nccl_uid; ncclGetUniqueId(&nccl_uid);
  // ncclComm_t comm; ncclCommInitRank(&comm, num_ranks, nccl_uid, rank);
  // c10d::NCCLComm nccl_comm(comm);

  ss << subSep << " TORCH NCCL version: " << c10d::getNcclVersion() << " " << subSep << std::endl;
  ss << subSep << " TORCH NCCL communicator for current device " << device << " " << subSep << std::endl;
  std::shared_ptr<c10d::NCCLComm> torchNCCLComm = pg->getNCCLComm();

  /** NOTE: when using the member functions defined in NCCLComm,
   * I run into an issue: undefined reference to `c10d::NCCLComm::getNcclComm()' 
   * later I've found out that all the member functions including`c10d::NCCLComm::getNcclComm()'
   * are local symbols that only visible inside the shared library `libtorch_cuda.so`,
   * with my own command as below:
   * nm /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so > libtorch_cuda.log
   * and the relevant output looks like:
   * 0000000000c98a40 t _ZN4c10d8NCCLComm11getNcclCommEv
   * 0000000000908e96 t _ZN4c10d8NCCLComm11getNcclCommEv.cold
   * thus we have no direct access to NCCLComm
   */
  // ncclComm_t ncclComm = torchNCCLComm->getNcclComm();
  // ncclUniqueId nccUID = torchNCCLComm->getNcclId(); auto ncclUIDString = getNCCLUIDString(nccUID);
  // ss << subSep << " NCCL communicator for current device " << device << " " << subSep << std::endl;
  // ss << ncclComm << " with NCCL ID: " << ncclUIDString << std::endl;

  logger.log(ss.str());
}


int main(int argc, char* argv[]) {
  // Print main arguments and dist-environment variables
  printMainArgs(argc, argv);
  printDistEnvVars();
  printProcessGroupBackend();

  // Parse launcher
  std::string launcher = parseLauncher(argc, argv);
  std::cout << "Launcher: " << launcher << std::endl;

  #ifndef USE_NCCL_AS_COMM_BACKEND
  TORCH_CHECK(launcher != "torchrun", "torchrun launcher is not supported for MPI backend");
  #endif

  // Creating Process Group
  #ifdef USE_NCCL_AS_COMM_BACKEND
    #ifdef USE_MY_NCCL_AS_COMM_BACKEND
      auto pg = createMyProcessGroupNCCL(argc, argv);
    #else
      auto pg = createProcessGroupNCCL(argc, argv);
    #endif
  #else
    // Init mpi process group ptr
    auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  #endif

  // Retrieving rank and num_ranks
  auto num_ranks = pg->getSize();
  auto rank = pg->getRank();

  // Init logger
  Logger logger("dist-mnist", rank);

  // Determine device
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "[Rank " << rank << "] " << "CUDA is available!" << std::endl;
    device = torch::Device(torch::kCUDA, rank); // put on cuda device with idx = rank
  }
  std::cout << "[Rank " << rank << "] " << "Running on device: " << device << std::endl;

  // Read train dataset
  const char* kDataRoot = "./data";
  auto train_dataset =
      torch::data::datasets::MNIST(kDataRoot)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());

  // Distributed Random Sampler
  auto data_sampler = torch::data::samplers::DistributedRandomSampler(
      train_dataset.size().value(), num_ranks, rank, false);

  auto num_train_samples_per_proc = train_dataset.size().value() / num_ranks;

  // Generate dataloader
  auto total_batch_size = 64;
  auto batch_size_per_proc = total_batch_size / num_ranks; // effective batch size in each processor
  auto data_loader = torch::data::make_data_loader(
    std::move(train_dataset), data_sampler, batch_size_per_proc
  );

  // Set manual seed
  torch::manual_seed(42);

  // Create model on CPU
  // here we can avoid explicit shared ptr creation thanks to TORCH_MODULE
  Model model; // auto model = std::make_shared<Model>();

  // Move model to determined device
  model->to(device);

  // Create optimizer
  auto learning_rate = 1e-2;
  torch::optim::SGD optimizer(model->parameters(), learning_rate);

  // Run training loop on train dataset
  size_t num_epochs = 10;
  int epoch_profile_start = 4, epoch_profile_stop = 6; // profile for epoch in range [epoch_profile_start, epoch_profile_stop)
  for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
    size_t num_correct = 0;

    if (epoch == epoch_profile_start) {
      cudaProfilerStart();
    }
    else if (epoch == epoch_profile_stop) {
      cudaProfilerStop();
    }

    std::string epoch_str = "Epoch " + std::to_string(epoch);
    auto rangeId = nvtxRangeStartA(epoch_str.c_str());
    for (auto& batch : *data_loader) {
      auto ip = batch.data.to(device);
      auto op = batch.target.squeeze().to(device);

      // convert to required formats
      ip = ip.to(torch::kF32);
      op = op.to(torch::kLong);

      // Reset gradients
      model->zero_grad();

      // Execute forward pass
      nvtxRangePushA("forward");
      auto prediction = model->forward(ip);
      nvtxRangePop();

      // Compute loss
      auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);

      // Backpropagation
      nvtxRangePushA("backward");
      loss.backward();
      nvtxRangePop();

      /** Averaging the gradients of the parameters in all the processors
       * NOTE: This may lag behind DistributedDataParallel (DDP) in performance
       * since this synchronizes parameters after backward pass while DDP
       * overlaps synchronizing parameters and computing gradients in backward pass 
       */
      nvtxRangePushA("grad allreduce");

      std::vector<::c10::intrusive_ptr<::c10d::Work>> works;
      for (auto& param : model->named_parameters()) {
        std::vector<at::Tensor> tmp = {param.value().grad()};
        auto work = pg->allreduce(tmp);
        works.push_back(std::move(work));
      }

      // wait the grad all-reduce to finish
      #ifdef USE_NCCL_AS_COMM_BACKEND
        waitWork<c10d::ProcessGroupNCCL>(pg, works);
      #else
        waitWork<c10d::ProcessGroupMPI>(pg, works);
      #endif

      for (auto& param : model->named_parameters()) {
        param.value().grad().data() = param.value().grad().data() / num_ranks;
      }

      nvtxRangePop();

      // Update parameters
      nvtxRangePushA("optimize");
      optimizer.step();
      nvtxRangePop();

      auto guess = prediction.argmax(1);
      num_correct += torch::sum(guess.eq_(op)).item<int64_t>();
    } // end batch loader
    nvtxRangeEnd(rangeId);

    // print accuracy for each epoch in each rank
    auto accuracy = 100.0 * num_correct / num_train_samples_per_proc;
    std::cout << "Accuracy in rank " << rank << " in epoch " << epoch << " - "
              << accuracy << std::endl;

  } // end epoch

  // Run Inference/Evaluation on test dataset only on rank 0
  if (rank == 0) {
    auto test_dataset =
        torch::data::datasets::MNIST(
            kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    auto num_test_samples = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader(
        std::move(test_dataset), num_test_samples);

    model->eval(); // enable eval mode to prevent backprop

    size_t num_correct = 0;

    for (auto& batch : *test_loader) {
      auto ip = batch.data.to(device);
      auto op = batch.target.squeeze().to(device);

      // convert to required format
      ip = ip.to(torch::kF32);
      op = op.to(torch::kLong);

      auto prediction = model->forward(ip);

      auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);

      std::cout << "Test loss - " << loss.item<float>() << std::endl;

      auto guess = prediction.argmax(1);

      num_correct += torch::sum(guess.eq_(op)).item<int64_t>();

    } // end test loader

    std::cout << "Num correct - " << num_correct << std::endl;
    std::cout << "Test Accuracy - " << 100.0 * num_correct / num_test_samples
              << std::endl;
  } // end rank 0


  // Analysis MyNCCL
  #ifdef USE_MY_NCCL_AS_COMM_BACKEND
    analysisMyNCCL(logger, pg, num_ranks, rank);
  #endif

  // finalize distributed environment
  #ifdef USE_NCCL_AS_COMM_BACKEND
    // stop all the threads and destroy all nccl comms until waiting all nccl kernels finished
    pg->shutdown();
    if (launcher == "mpirun")
      // explicitly finalize MPI if using mpirun
      MPI_Finalize();
    #ifdef USE_MY_NCCL_AS_COMM_BACKEND
      std::cout << "[Rank " << rank << "] " << "Distributed environment finalized for MyNCCL" << std::endl;
    #else
      std::cout << "[Rank " << rank << "] " << "Distributed environment finalized for NCCL" << std::endl;
    #endif 
  #else
    // // stop all the threads and MPI comms
    // pg->abort();
    // MPI_Finalize();
    std::cout << "[Rank " << rank << "] " << "Distributed environment finalized for MPI" << std::endl;
  #endif

  std::cout << "[Rank " << rank << "] " << "Training ends successfully" << std::endl;

  return 0;
}
