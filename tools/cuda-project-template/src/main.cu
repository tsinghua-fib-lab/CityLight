/**
 * @file main.cu
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date YYYY-MM-DD
 *
 * @copyright Copyright (c) 2021 Tsingroc
 *
 */

#include <cuda.h>
#include <iostream>

__global__ void cuda_hello() {
  printf("Hello World from GPU!\n");
  return;
}

/**
 * @brief 主函数
 *
 * @param argc 参数数量
 * @param argv 参数值
 * @return int 程序返回值
 */
int main(int argc, char** argv) {
  cuda_hello<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}
