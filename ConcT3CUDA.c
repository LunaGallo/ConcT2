#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCO 5
#define NPART 4
#define EXEC_NUM 10

void CudaSmooth(unsigned char *CpuInput, unsigned char *CpuOutput, int largura, int altura);

//As estruturas a seguir foram criadas para representar uma imagem PPM seja ela P5 ou P6
//PPMGrayPixel representa um pixel em P5 , ou seja, em escala de cinza de 0 a 255.
typedef struct {
     unsigned char value;
} PPMGrayPixel;

//PPMPixel representa um pixel em P6 , ou seja, em RGB com 3 canais variando de 0 a 255.
typedef struct {
     unsigned char red,green,blue;
} PPMPixel;


//PPMImage representa uma imagem P5 ou P6. Inclui a possibilidade de armazenar bordas superior e inferior de pixeis redundantes(especifico para a aplicação).
typedef struct {
     int x, y;//dimensões da imagem.
     PPMPixel *data;//vetor de pixeis, caso seja uma imagem colorida. NULL caso contrario
     PPMGrayPixel *grayData;//vetor de pixeis, caso seja uma imagem cinza. NULL caso contrario
} PPMImage;

//-----------------------------
//Código com funções para ler e escrever imagens PPM em arquivo
#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255
static PPMImage *readPPM(const char *filename){
         char buff[16];
         PPMImage *img;
         FILE *fp;
         int c, rgb_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

         //read image format
         if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

    //check the image format
    if (buff[0] != 'P' || (buff[1] != '6' && buff[1] != '5')) {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }

    //alloc memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    if (buff[1] == '6'){
		img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

		img->grayData = NULL;

		if (!img) {
			 fprintf(stderr, "Unable to allocate memory\n");
			 exit(1);
		}

		//read pixel data from file
		if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
			 fprintf(stderr, "Error loading image '%s'\n", filename);
			 exit(1);
		}
	}
	else if (buff[1] == '5'){
		img->data = NULL;

		img->grayData = (PPMGrayPixel*)malloc(img->x * img->y * sizeof(PPMGrayPixel));

		if (!img) {
			 fprintf(stderr, "Unable to allocate memory\n");
			 exit(1);
		}

		//read pixel data from file
		if (fread(img->grayData, 1 * img->x, img->y, fp) != img->y) {
			 fprintf(stderr, "Error loading image '%s'\n", filename);
			 exit(1);
		}
	}

    fclose(fp);
    return img;
}
void writePPM(const char *filename, PPMImage *img){
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //comments
    fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}
//---------------------------------

/*
A seguinte função escreve no arquivo "res.ppm" a imagem colorida "imagem" dada.
*/
void writeColoredPPMImage(PPMImage *imagem, FILE *arquivo) {
	int i, j, largura, altura;

	largura = imagem->x;
	altura = imagem->y;

	arquivo = fopen("res.ppm", "w");
	fprintf(arquivo, "P3\n%d %d\n255\n", largura, altura);

	for (i = 0; i<altura; i++) {
		for (j = 0; j<largura; j++) {
			fprintf(arquivo, "%d ", imagem->data[((i)*(largura)) + (j)].red);
			fprintf(arquivo, "%d ", imagem->data[((i)*(largura)) + (j)].green);
			fprintf(arquivo, "%d   ", imagem->data[((i)*(largura)) + (j)].blue);
		}
		fprintf(arquivo, "\n");
	}

	fclose(arquivo);
/*
	arquivo = fopen("ori.ppm", "w");
	fprintf(arquivo, "P3\n%d %d\n255\n", largura, altura);

	for (i = 0; i<altura; i++) {
		for (j = 0; j<largura; j++) {
			fprintf(arquivo, "%d ", imagem->data[((i)*(largura)) + (j)].red);
			fprintf(arquivo, "%d ", imagem->data[((i)*(largura)) + (j)].green);
			fprintf(arquivo, "%d   ", imagem->data[((i)*(largura)) + (j)].blue);
		}
		fprintf(arquivo, "\n");
	}

	fclose(arquivo);
*/
}

/*

*/


/*
A seguinte função escreve no arquivo "res.ppm" a imagem cinza "imagem" dada.
*/
void writeGrayscalePPMImage(PPMImage *imagem, FILE *arquivo) {
	int i, j, largura, altura;

	largura = imagem->x;
	altura = imagem->y;

    arquivo = fopen("res.ppm", "w");
    fprintf(arquivo, "P2\n%d %d\n255\n", largura, altura);

    for(i=0; i<altura; i++){
        for(j=0; j<largura; j++){
            fprintf(arquivo,"%d "  ,imagem->grayData[((i)*(largura))+(j)].value);
        }
        fprintf(arquivo, "\n");
    }

    fclose(arquivo);

/*
    arquivo = fopen("ori.ppm", "w");
    fprintf(arquivo, "P2\n%d %d\n255\n", largura, altura);

    for(i=0; i<altura; i++){
        for(j=0; j<largura; j++){
            fprintf(arquivo,"%d "  ,imagem->grayData[((i)*(largura))+(j)].value);
        }
        fprintf(arquivo, "\n");
    }

    fclose(arquivo);
*/
}

/*
A seguinte função retorna uma imagem alocada na memória, criada com base numa 
imagem dada "imagemFonte" mas sem copiar seu conteudo, somente dimensões e bordas.
*/
PPMImage* CreateBasedPPMImage(PPMImage *imagemFonte){
	PPMImage *imagemResult;
    if(imagemFonte == NULL){
		return NULL;
	}
	imagemResult = (PPMImage*) malloc(sizeof(PPMImage));
	imagemResult->x = imagemFonte->x;
	imagemResult->y = imagemFonte->y;
	if(imagemFonte->data != NULL){ //Colored Image
		imagemResult->data = (PPMPixel*) malloc((imagemFonte->x)*(imagemFonte->y) * sizeof(PPMPixel));
		imagemResult->grayData = NULL;
	}
	else{ //Grayscale Image
		imagemResult->grayData = (PPMGrayPixel*) malloc((imagemFonte->x)*(imagemFonte->y) * sizeof(PPMGrayPixel));
		imagemResult->data = NULL;
	}

	return imagemResult;
}

/*
A seguinte função libera o espaço da memória previamente alocado para 
a imagem cujo endereço está no ponteiro "imageReference".
*/
void DestroyPPMImage(PPMImage **imageReference){
    if((*imageReference) == NULL){
		return;
	}
	if((*imageReference)->data != NULL){ //Colored Image
        free((*imageReference)->data);
	}
	else{ //Grayscale Image
        free((*imageReference)->grayData);

	}
	free(*imageReference);
	(*imageReference) = NULL;
}



int main(){
    FILE* arquivo = NULL;
    int i, j, k, channelNum;

	//Imagens e vetores de imagens
    PPMImage *imagemFonte;
    PPMImage *imagemFinal;
    unsigned char **ImgInput;
    unsigned char **ImgOutput;

	//Variáveis de contagem de tempo
	struct timespec startingTimespec, endingTimespec;
	long long startingTime, totalTime;
	double  timeSum;
	
	//Loop de repetição de execussão, para extrair a média dos tempos de execussão
	for(j=0;j<EXEC_NUM;j++){
		
		//Le o arquivo e divide a imagem
		imagemFonte = readPPM("img.ppm");
		
		if(imagemFonte->data != NULL){ //Colored Image
			channelNum = 3;
		}
		else{ //Grayscale Image
			channelNum = 1;
		}
		ImgInput = (unsigned char **) malloc (channelNum * sizeof(unsigned char*));
		ImgOutput = (unsigned char **) malloc (channelNum * sizeof(unsigned char*));
		for(i=0;i<channelNum; i++){
			ImgInput[i] = (unsigned char *) malloc ((imagemFonte->x * imagemFonte->y) * sizeof(unsigned char));
			ImgOutput[i] = (unsigned char *) malloc ((imagemFonte->x * imagemFonte->y) * sizeof(unsigned char));
		}
		
		if(imagemFonte->data != NULL){ //Colored Image
			for(k=0;k<(imagemFonte->x * imagemFonte->y); k++){
				ImgInput[0][k] = imagemFonte->data[k].red;
				ImgInput[1][k] = imagemFonte->data[k].green;
				ImgInput[2][k] = imagemFonte->data[k].blue;
			}
		}
		else{ //Grayscale Image
			for(k=0;k<(imagemFonte->x * imagemFonte->y); k++){
				ImgInput[0][k] = imagemFonte->grayData[k].value;
			}
		}
		
		//Começa a contar o tempo
		clock_gettime(CLOCK_REALTIME, &startingTimespec);
		startingTime = startingTimespec.tv_sec*1000000000LL + startingTimespec.tv_nsec;
		
		for(i=0;i<channelNum; i++){
			CudaSmooth(ImgInput[i], ImgOutput[i], imagemFonte->x, imagemFonte->y);
		}
		
		//Para de contar o tempo
		clock_gettime(CLOCK_REALTIME, &endingTimespec);
		totalTime = endingTimespec.tv_sec*1000000000LL + endingTimespec.tv_nsec - startingTime;
		double miliseconds = totalTime/1000000;
		timeSum += (miliseconds/1000);

		//Cria a imagem final e 
		imagemFinal = CreateBasedPPMImage(imagemFonte);
		if(imagemFonte->data != NULL){ //Colored Image
			for(k=0;k<(imagemFonte->x * imagemFonte->y); k++){
				imagemFinal->data[k].red = ImgOutput[0][k];
				imagemFinal->data[k].green = ImgOutput[1][k];
				imagemFinal->data[k].blue = ImgOutput[2][k];
			}
		}
		else{ //Grayscale Image
			for(k=0;k<(imagemFonte->x * imagemFonte->y); k++){
				imagemFinal->grayData[k].value = ImgOutput[0][k];
			}
		}
		

		//Escreve a imagem final no arquivo
		if(imagemFinal->data != NULL){ //Colored Image
			writeColoredPPMImage(imagemFinal, arquivo);
		}
		else{ //Grayscale Image
			writeGrayscalePPMImage(imagemFinal, arquivo);
		}

		//Libera a memória de todas as imagens usadas
		for (i=0; i<NPART; i++) {
			DestroyPPMImage(&(imagemVector[i]));
			DestroyPPMImage(&(imagemResults[i]));
		}
		DestroyPPMImage(imagemVector);
		DestroyPPMImage(imagemResults);
		DestroyPPMImage(&imagemFonte);
		DestroyPPMImage(&imagemFinal);

	}
	//Imprime o tempo médio de execução
	printf("Avarage Execution Time = %f\n\n", timeSum/EXEC_NUM);

    return 0;
}
