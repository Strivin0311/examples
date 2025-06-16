/** 
 * NOTE: we've modified the source code according to this pr: https://github.com/pytorch/examples/pull/1341
*/
#ifndef USE_C10D_MPI
#define USE_C10D_MPI
#endif

#include <iostream>
#include <string>

#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/torch.h>

#include <cuda_profiler_api.h>
#include "nvtx3/nvToolsExt.h"

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

void waitWork(
    c10::intrusive_ptr<c10d::ProcessGroupMPI> pg,
    std::vector<c10::intrusive_ptr<c10d::Work>> works) {
  for (auto& work : works) {
    try {
      work->wait();
    } catch (const std::exception& ex) {
      std::cerr << "Exception received: " << ex.what() << std::endl;
      pg->abort();
    }
  }
}

int main(int argc, char* argv[]) {
  // Creating MPI Process Group
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();

  // Retrieving MPI environment variables
  auto numranks = pg->getSize();
  auto rank = pg->getRank();

  // Determine device
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "[Rank " << rank << "] " << "CUDA is available! Training on GPU " << rank << std::endl;
    device = torch::Device(torch::kCUDA, rank); // put on cuda device with idx = rank
  }

  // Read train dataset
  const char* kDataRoot = "./data";
  auto train_dataset =
      torch::data::datasets::MNIST(kDataRoot)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());

  // Distributed Random Sampler
  auto data_sampler = torch::data::samplers::DistributedRandomSampler(
      train_dataset.size().value(), numranks, rank, false);

  auto num_train_samples_per_proc = train_dataset.size().value() / numranks;

  // Generate dataloader
  auto total_batch_size = 64;
  auto batch_size_per_proc = total_batch_size / numranks; // effective batch size in each processor
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
      waitWork(pg, works);

      for (auto& param : model->named_parameters()) {
        param.value().grad().data() = param.value().grad().data() / numranks;
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
}
