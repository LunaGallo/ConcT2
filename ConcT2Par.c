#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

#define BLOCO 5
#define NPART 4
#define EXEC_NUM 10

typedef struct {
     unsigned char value;
} PPMGrayPixel;

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
     PPMGrayPixel *grayData;

     PPMPixel *ColTopBorder;
     PPMGrayPixel *GrayTopBorder;
     PPMPixel *ColBotBorder;
     PPMGrayPixel *GrayBotBorder;
} PPMImage;

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


    img->ColTopBorder = NULL;
    img->GrayTopBorder = NULL;
    img->ColBotBorder = NULL;
    img->GrayBotBorder = NULL;

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


void ColoredSmooth(PPMImage *imagemFonte, PPMImage *imagemResult){
    int largura, altura;
    int i, j, k, l, contador, R, G, B;

	largura = imagemFonte->x;
	altura = imagemFonte->y;

#pragma omp parallel for collapse(2) private(contador, i, j, k, l, R, G, B)
    for(i=0;i<altura;i++){
        for(j=0;j<largura;j++){
            R = 0;
            G = 0;
            B = 0;
            contador = 0;
            for(k=-((BLOCO-1)/2); k<=((BLOCO-1)/2); k++){
                for(l=-((BLOCO-1)/2); l<=((BLOCO-1)/2); l++){
                    if ((j + l >= 0) && (j + l < largura)) {
                        if (i + k >= 0) {
                            if(i + k < altura) {
                                R += imagemFonte->data[((i+k)*(largura))+(j+l)].red;
                                G += imagemFonte->data[((i+k)*(largura))+(j+l)].green;
                                B += imagemFonte->data[((i+k)*(largura))+(j+l)].blue;
                                contador++;
                            }
                            else if((imagemFonte->ColBotBorder != NULL)) {
                                R += imagemFonte->ColBotBorder[((i+k)*(largura))+(j+l)-(imagemFonte->y*imagemFonte->x)].red;
                                G += imagemFonte->ColBotBorder[((i+k)*(largura))+(j+l)-(imagemFonte->y*imagemFonte->x)].green;
                                B += imagemFonte->ColBotBorder[((i+k)*(largura))+(j+l)-(imagemFonte->y*imagemFonte->x)].blue;
                                contador++;
                            }
                        }
                        else if((imagemFonte->ColTopBorder != NULL)){
                            R += imagemFonte->ColTopBorder[((i+k)*(largura))+(j+l)+ (BLOCO*imagemFonte->x)].red;
                            G += imagemFonte->ColTopBorder[((i+k)*(largura))+(j+l)+ (BLOCO*imagemFonte->x)].green;
                            B += imagemFonte->ColTopBorder[((i+k)*(largura))+(j+l)+ (BLOCO*imagemFonte->x)].blue;
                            contador++;
                        }
                    }
                }
            }
            imagemResult->data[((i)*(largura))+(j)].red = (unsigned char) (R / contador);
            imagemResult->data[((i)*(largura))+(j)].green = (unsigned char) (G / contador);
            imagemResult->data[((i)*(largura))+(j)].blue = (unsigned char) (B / contador);
        }
    }

}

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

void GrayscaleSmooth(PPMImage *imagemFonte, PPMImage *imagemResult) {
	int largura, altura;
	int i, j, k, l, contador, V;

	largura = imagemFonte->x;
	altura = imagemFonte->y;

#pragma omp parallel for collapse(2) private(contador, i, j, k, l, V)
	for (i = 0; i < altura; i++) {
		for (j = 0; j < largura; j++) {
			V = 0;

			contador = 0;
			for (k = -((BLOCO - 1) / 2); k <= ((BLOCO - 1) / 2); k++) {
				for (l = -((BLOCO - 1) / 2); l <= ((BLOCO - 1) / 2); l++) {
                    if ((j + l >= 0) && (j + l < largura)) {
                        if (i + k >= 0) {
                            if(i + k < altura) {
                                V += imagemFonte->grayData[((i + k)*(largura)) + (j + l)].value;
                                contador++;
                            }
                            else if((imagemFonte->GrayBotBorder != NULL)) {
                                V += imagemFonte->GrayBotBorder[((i+k)*(largura))+(j+l)-(imagemFonte->y*imagemFonte->x)].value;
                                contador++;
                            }
                        }
                        else if((imagemFonte->GrayTopBorder != NULL)){
                            V += imagemFonte->GrayTopBorder[((i + k)*(largura)) + (j + l) + (BLOCO*imagemFonte->x)].value;
                            contador++;
                        }
                    }
				}
			}
			imagemResult->grayData[((i)*(largura)) + (j)].value = (unsigned char)(V / contador);
		}
	}
}

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


	if(imagemFonte->ColTopBorder == NULL){
        imagemResult->ColTopBorder = NULL;
    }
    else{
        imagemResult->ColTopBorder = (PPMPixel*) malloc((imagemFonte->x)*(BLOCO) * sizeof(PPMPixel));
    }

	if(imagemFonte->GrayTopBorder == NULL){
        imagemResult->GrayTopBorder = NULL;
    }
    else{
        imagemResult->GrayTopBorder = (PPMGrayPixel*) malloc((imagemFonte->x)*(BLOCO) * sizeof(PPMGrayPixel));
    }

	if(imagemFonte->ColBotBorder == NULL){
        imagemResult->ColBotBorder = NULL;
    }
    else{
        imagemResult->ColBotBorder = (PPMPixel*) malloc((imagemFonte->x)*(BLOCO) * sizeof(PPMPixel));
    }

	if(imagemFonte->GrayBotBorder == NULL){
        imagemResult->GrayBotBorder = NULL;
    }
    else{
        imagemResult->GrayBotBorder = (PPMGrayPixel*) malloc((imagemFonte->x)*(BLOCO) * sizeof(PPMGrayPixel));
    }
	return imagemResult;
}

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

PPMImage** CropPPMImage(PPMImage *imagemFonte) {
    int i, j, globalIndex, localIndex;
    PPMImage **imageVector = (PPMImage**)malloc((NPART) * sizeof(PPMImage*));

    globalIndex = 0;
    for (i=0; i<NPART; i++) {
        imageVector[i] = (PPMImage*)malloc(sizeof(PPMImage));
        imageVector[i]->x = imagemFonte->x;
        imageVector[i]->y = imagemFonte->y/NPART;
        if (i == NPART-1) {
            imageVector[i]->y += imagemFonte->y%NPART;
        }



        if(imagemFonte->data != NULL){ //Colored Image


            imageVector[i]->data = (PPMPixel*) malloc((imageVector[i]->x)*(imageVector[i]->y) * sizeof(PPMPixel));
            imageVector[i]->grayData = NULL;

            imageVector[i]->GrayBotBorder = NULL;
            imageVector[i]->GrayTopBorder = NULL;
            if(i==0){
                imageVector[i]->ColTopBorder = NULL;
            }
            else{
                imageVector[i]->ColTopBorder = (PPMPixel*) malloc((imageVector[i]->x)*(BLOCO) * sizeof(PPMPixel));
            }
            if(i==NPART-1){
                imageVector[i]->ColBotBorder = NULL;
            }
            else{
                imageVector[i]->ColBotBorder = (PPMPixel*) malloc((imageVector[i]->x)*(BLOCO) * sizeof(PPMPixel));
            }

            if(i > 0){
                for (j=0, localIndex=-(BLOCO * imageVector[i]->x); localIndex<0 ; j++, localIndex++) {
                    imageVector[i]->ColTopBorder[j] = imagemFonte->data[globalIndex + localIndex];
                }
            }

            for(j=0; j<(imageVector[i]->y * imageVector[i]->x) ; j++, globalIndex++){
                imageVector[i]->data[j] = imagemFonte->data[globalIndex];
            }

            if(i < NPART-1){
                for (j=0, localIndex=0; localIndex<(BLOCO * imageVector[i]->x) ; j++, localIndex++) {
                    imageVector[i]->ColBotBorder[j] = imagemFonte->data[globalIndex + localIndex];
                }
            }

        }



        else{//Grayscale Image


            imageVector[i]->data = NULL;
            imageVector[i]->grayData = (PPMGrayPixel*) malloc((imageVector[i]->x)*(imageVector[i]->y) * sizeof(PPMGrayPixel));


            if(i==0){
                imageVector[i]->GrayTopBorder = NULL;
            }
            else{
                imageVector[i]->GrayTopBorder = (PPMGrayPixel*) malloc((imageVector[i]->x)*(BLOCO) * sizeof(PPMGrayPixel));
            }

            if(i==NPART-1){
                imageVector[i]->GrayBotBorder = NULL;
            }
            else{
                imageVector[i]->GrayBotBorder = (PPMGrayPixel*) malloc((imageVector[i]->x)*(BLOCO) * sizeof(PPMGrayPixel));
            }

            imageVector[i]->ColBotBorder = NULL;
            imageVector[i]->ColTopBorder = NULL;


            if(i > 0){
                for (j=0, localIndex=-(BLOCO * imageVector[i]->x); localIndex<0 ; j++, localIndex++) {
                    imageVector[i]->GrayTopBorder[j] = imagemFonte->grayData[globalIndex + localIndex];
                }
            }

            for(j=0; j<(imageVector[i]->y * imageVector[i]->x) ; j++, globalIndex++){
                imageVector[i]->grayData[j] = imagemFonte->grayData[globalIndex];
            }

            if(i < NPART-1){
                for (j=0, localIndex=0; localIndex<(BLOCO * imageVector[i]->x) ; j++, localIndex++) {
                    imageVector[i]->GrayBotBorder[j] = imagemFonte->grayData[globalIndex + localIndex];
                }
            }
        }

    }

    return imageVector;
}

void MergePPMImages(PPMImage *mergedImage, PPMImage **imageVector) {
    int i, j, globalIndex;

    globalIndex = 0;
    for (i=0; i<NPART; i++) {
        if(mergedImage->data != NULL){ //Colored Image
            for(j=0; j<(imageVector[i]->y * imageVector[i]->x) ; j++, globalIndex++){
                mergedImage->data[globalIndex] = imageVector[i]->data[j];
            }
        }
        else{//Grayscale Image
            for(j=0; j<(imageVector[i]->y * imageVector[i]->x) ; j++, globalIndex++){
                mergedImage->grayData[globalIndex] = imageVector[i]->grayData[j];
            }
        }
    }
}

int main(){
    FILE* arquivo = NULL;
    int i, j;

    PPMImage *imagemFonte;
    PPMImage *imagemFinal;
	PPMImage **imagemResults;
	PPMImage **imagemVector;

	struct timespec startingTimespec, endingTimespec;
	long long startingTime, totalTime;
	double  timeSum;
	
	for(j=0;j<EXEC_NUM;j++){
		
		//Starts counting time
		clock_gettime(CLOCK_REALTIME, &startingTimespec);
		startingTime = startingTimespec.tv_sec*1000000000LL + startingTimespec.tv_nsec;

		imagemFonte = readPPM("img.ppm");
		imagemVector = CropPPMImage(imagemFonte);
		imagemResults = (PPMImage**)malloc(NPART * sizeof(PPMImage*));
		

		for (i=0; i<NPART; i++) {
			imagemResults[i] = CreateBasedPPMImage(imagemVector[i]);

			if(imagemResults[i]->data != NULL){ //Colored Image
				ColoredSmooth(imagemVector[i], imagemResults[i]);
			}
			else{ //Grayscale Image
				GrayscaleSmooth(imagemVector[i], imagemResults[i]);
			}
		}

		imagemFinal = CreateBasedPPMImage(imagemFonte);
		MergePPMImages(imagemFinal, imagemResults);
		
		{
			//Stops counting time
			clock_gettime(CLOCK_REALTIME, &endingTimespec);
			totalTime = endingTimespec.tv_sec*1000000000LL + endingTimespec.tv_nsec - startingTime;
			double miliseconds = totalTime/1000000;
			timeSum += (miliseconds/1000);
		}

		if(imagemFinal->data != NULL){ //Colored Image
			writeColoredPPMImage(imagemFinal, arquivo);
		}
		else{ //Grayscale Image
			writeGrayscalePPMImage(imagemFinal, arquivo);
		}

		for (i=0; i<NPART; i++) {
			DestroyPPMImage(&(imagemVector[i]));
			DestroyPPMImage(&(imagemResults[i]));
		}
		DestroyPPMImage(imagemVector);
		DestroyPPMImage(imagemResults);
		DestroyPPMImage(&imagemFonte);
		DestroyPPMImage(&imagemFinal);

	}
	printf("Avarage Execution Time = %f\n\n", timeSum/EXEC_NUM);

    return 0;
}
