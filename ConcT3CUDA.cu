extern "C"
{
#include "ConcT3CUDA.h"
}


__global__ void SingleSmooth_Kernel(unsigned char *imagemResult, unsigned char *imagemFonte, int largura, int altura){
	int i, j, k, l, contador, V;

	i = threadIdx.y;
	j = threadIdx.x;
	V = 0;
	
	contador = 0;
	for (k = -((BLOCO - 1) / 2); k <= ((BLOCO - 1) / 2); k++) {
		for (l = -((BLOCO - 1) / 2); l <= ((BLOCO - 1) / 2); l++) {
			if ((j + l >= 0) && (j + l < largura)) {
				if ((i + k >= 0) && (i + k < altura)) {
					V += imagemFonte[((i + k)*(largura)) + (j + l)];
					contador++;
				}
			}
		}
	}
	imagemResult[((i)*(largura)) + (j)] = (unsigned char)(V / contador);
}


void CudaSmooth(unsigned char *CpuInput, unsigned char *CpuOutput, int largura, int altura) {
	unsigned char *GpuInput;
	unsigned char *GpuOutput;
	
	cudaMalloc(&GpuInput, largura*altura);
	cudaMalloc(&GpuOutput, largura*altura);

	cudaMemcpy(GpuInput, CpuInput, largura*altura, cudaMemcpyHostToDevice);

	dim3 block_size(64, 64);

	dim3 grid_size;
	grid_size.x = (largura + block_size.x - 1)/block_size.x;
	grid_size.y = (altura + block_size.y - 1)/block_size.y;

	SingleSmooth_Kernel<<<grid_size, block_size>>>(GpuInput, GpuOutput, largura, altura);

	cudaMemcpy(CpuOutput, GpuOutput, largura*altura, cudaMemcpyDeviceToHost);

	cudaFree(GpuInput);
	cudaFree(GpuOutput);
}
