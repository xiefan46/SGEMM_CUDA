#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 12) {
    std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

  // print some device info
  CudaDeviceInfo();



  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS FLOPs ceiling is reached at 8192
  // 加入两个矩形的测试
  std::vector<int> M_SIZE = {128, 256, 512, 1024, 2048, 4096, 2048, 4096, 1033};
  std::vector<int> N_SIZE = {128, 256, 512, 1024, 2048, 4096, 4096, 1024, 899};
  std::vector<int> K_SIZE = {128, 256, 512, 1024, 2048, 4096, 1024, 2048, 671};

  float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C


  int repeat_times = 50;
  for (int i = 0; i < M_SIZE.size(); i++) {
    long m = M_SIZE[i], n = N_SIZE[i], k = K_SIZE[i];
    std::cout << "m size: " << m << " n size: "<< n << " k size: "<< k << std::endl;

    // Declare the handle, create the handle, cublasCreate will return a value of
  	// type cublasStatus_t to determine whether the handle was created
  	// successfully (the value is 0)
  	cublasHandle_t handle;
  	if (cublasCreate(&handle)) {
    	std::cerr << "Create cublas handle error." << std::endl;
    	exit(EXIT_FAILURE);
  	};

    float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr; // host matrices
  	float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices

  	A = (float *)malloc(sizeof(float) * m * k);
  	B = (float *)malloc(sizeof(float) * n * k);
  	C = (float *)malloc(sizeof(float) * m * n);
  	C_ref = (float *)malloc(sizeof(float) * m * n);

  	randomize_matrix(A, m * k);
  	randomize_matrix(B, n * k);
  	randomize_matrix(C, m * n);

  	cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * m * k));
  	cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * n * k));
  	cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * m * n));
  	cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * m * n));

  	cudaCheck(cudaMemcpy(dA, A, sizeof(float) * m * k,
                       cudaMemcpyHostToDevice));
  	cudaCheck(cudaMemcpy(dB, B, sizeof(float) * n * k,
                       cudaMemcpyHostToDevice));
  	cudaCheck(cudaMemcpy(dC, C, sizeof(float) * m * n,
                       cudaMemcpyHostToDevice));
  	cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * m * n,
                       cudaMemcpyHostToDevice));


   	// std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              // << ", beta: " << beta << std::endl;
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0) {
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
                 handle); // cuBLAS
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                 handle); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

      if (!verify_matrix(C_ref, C, m * n)) {
        std::cout
            << "Failed to pass the correctness verification against NVIDIA "
               "cuBLAS."
            << std::endl;
        if (m <= 128) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix(A, m, n, fs);
          fs << "B:\n";
          print_matrix(B, m, n, fs);
          fs << "C:\n";
          print_matrix(C, m, n, fs);
          fs << "Should:\n";
          print_matrix(C_ref, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }
    }

    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      // We don't reset dC between runs to save time
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. m size: "
        "(%ld). n size: (%ld), k size: (%ld) \n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m, n, k);
    fflush(stdout);
    // make dC and dC_ref equal again (we modified dC while calling our kernel
    // for benchmarking)
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                         cudaMemcpyDeviceToDevice));

    // Free up CPU and GPU space
  	free(A);
  	free(B);
  	free(C);
  	free(C_ref);
  	cudaFree(dA);
  	cudaFree(dB);
  	cudaFree(dC);
  	cudaFree(dC_ref);
  	cublasDestroy(handle);
  }
  return 0;
};