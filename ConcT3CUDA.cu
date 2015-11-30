#define BLOCO 5

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
		
