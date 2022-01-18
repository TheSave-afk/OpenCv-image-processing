//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>

#define _USE_MATH_DEFINES
#include <cmath>

struct ArgumentList {
	std::string image_name;		    //!< image file name
	int wait_t;                     //!< waiting time
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {

	if(argc<3 || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help"))
	{
		std::cout<<"usage: simple -i <image_name>"<<std::endl;
		std::cout<<"exit:  type q"<<std::endl<<std::endl;
		std::cout<<"Allowed options:"<<std::endl<<
				"   -h	                     produce help message"<<std::endl<<
				"   -i arg                   image name. Use %0xd format for multiple images."<<std::endl<<
				"   -t arg                   wait before next frame (ms) [default = 0]"<<std::endl<<std::endl<<std::endl;
		return false;
	}

	int i = 1;
	while(i<argc)
	{
		if(std::string(argv[i]) == "-i") {
			args.image_name = std::string(argv[++i]);
		}

		if(std::string(argv[i]) == "-t") {
			args.wait_t = atoi(argv[++i]);
		}
		else
			args.wait_t = 0;

		++i;
	}

	return true;
}

//maxpooling
void maxPooling(const cv::Mat& image, int size, int stride, cv::Mat & out)
{
	//massimo temporaneo
	int max_t;
	//ciclo l'immagine originale e salvo i valori nel kernel_vector
	for(int r = 0; r<image.rows-size+1; r+=stride)
	{
		for(int c = 0; c<image.cols-size+1; c+=stride)
		{

			max_t = 0;
			//ciclo per il kernel
			for (int kr=0; kr < size; kr++)
			{
				for(int kc=0; kc < size; kc++)
				{
					if(image.data[((kr + r) * image.cols + (c + kc))] > max_t)
					{
						max_t =image.data[((kr + r)*image.cols + (c + kc))];
					}
				}

			out.data[(r/stride)*out.cols + (c/stride)] = max_t;
			}
		}
	}
}

//averagePooling
void averagePooling(const cv::Mat& image, int size, int stride, cv::Mat & out)
{
	unsigned int avg;
	unsigned int somma;

	for (int r = 0 ; r < image.rows-size+1; r += stride)
	{
		for(int c = 0; c <image.cols-size+1; c += stride)
		{

			somma = 0;
			//ciclo per il kernel
			for (int kr=0; kr < size; kr++)
			{
				for(int kc=0; kc < size; kc++)
				{
					somma += image.data[((r + kr)*image.cols + (c + kc))];
				}
			}

			avg =(somma/(size*size));
			out.data[(r/stride)*out.cols + (c/stride)] = avg;
		}
		avg = 0;
	}
}

//convoluzione
void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out, int stride=1)
{
	float conv;
	int a = floor((kernel.rows -1)/2);
	int b = floor((kernel.cols -1)/2);

	for(int r = 0; r < image.rows; r += stride)
	{
		for(int c = 0; c < image.cols; c += stride)
		{
			if( (r-a)<0 || (c-b)<0 || (r+a)>image.rows -1 || (c+b)>image.cols -1 )
			{
				*((float*)&out.data[((c/stride) + (r/stride)*out.cols) *out.elemSize()]) = 0;
			}
			else
			{
				conv = 0.0;
				for(int kr = -a; kr < kernel.rows -a; kr ++)
				{
					for(int kc = -b; kc < kernel.cols - b; kc ++)
					{
						conv += *((float*)&kernel.data[((b + kc) + (a + kr)*kernel.cols)* kernel.elemSize()]) * ((float)(image.data[((r + kr)*image.cols + (c + kc))*image.elemSize()]));
					}
				}

				*((float*)&out.data[((r/stride)*out.cols + (c/stride)) * out.elemSize()]) = conv;
			}
		}
	}
}

//convoluzione interi
void conv(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out, int stride=1)
{
	cv::Mat processingFloat = cv::Mat::zeros(out.rows, out.cols, CV_32FC1);
	//richiamo la funzione convoluzione float ed utilizzo la matrice di uscita per il processing successivo
	convFloat(image,kernel,processingFloat);

	//dichiaro i valori di massimo e minimo da utilizzare per il contrast stretching
	float massimo=0;
	float minimo=0;

	//trovo il massimo ed il minimo scorrendo l'immagine float
	for(int r = 0; r < processingFloat.rows; r++)
	{
		for(int c = 0; c < processingFloat.cols; c++)
		{
			//puntatore a float della matrice convFloat
			float *ptr = ((float *)&processingFloat.data[(r*processingFloat.cols + c)*processingFloat.elemSize()]);

			if (*ptr > massimo)
			{
				massimo = *ptr;
			}
			if(*ptr < minimo)
			{
				minimo = *ptr;
			}
		}
	}

	for(int r = 0; r < out.rows; r++)
	{
		for(int c = 0; c < out.cols; c++)
		{
			float *ptr = ((float *)&processingFloat.data[(r*processingFloat.cols + c)*processingFloat.elemSize()]);
			unsigned int norm = (*ptr - minimo) * 255/(massimo - minimo);
			out.data[(r*out.cols + c) * out.elemSize()] = norm;
		}
	}
}

//Funzione per effettuare il contrast stretching nelle matrici
// corrisponde al codice presente nella parte finale della convoluzione intera
void normalizeMatrix(cv::Mat& sourceMatrix, cv::Mat& normalized)
{
	float min = 0;
	float max = 0;

	for(int r = 0; r < sourceMatrix.rows;r++)
	{
		for(int c=0; c < sourceMatrix.cols;c++)
		{
			float *ptr = ((float*)&sourceMatrix.data[(c + r*sourceMatrix.cols) * sourceMatrix.elemSize()]);
			if(*ptr > max)
			{
				max = *ptr;
			}
			if(*ptr < min)
			{
				min = *ptr;
			}
		}
	}

	for(int r=0; r < normalized.rows; r++)
	{
		for(int c=0; c < normalized.cols; c++)
		{
			//puntatore della matrice normalizzata
			float *ptr = ((float*)&sourceMatrix.data[(c + r*sourceMatrix.cols)*sourceMatrix.elemSize()]);
			//contrast stretching secondo la formula
			normalized.data[c + r*normalized.cols] = ((*ptr - min) * 255/(max - min));
		}
	}
}

//kernel gaussiano 1-D
void gaussianKernel(float sigma, int radius, cv::Mat& kernel)
{
	float sum =0;
	for(int c = -radius; c <= radius; c++)
	{
		//dichiaro la variabile gaussiano che consiste nella formula
		float gaussiano = exp(-((c*c)/(2*sigma*sigma))/(2 *M_PI*sigma*sigma));
		//puntatore per la matrice kernel
		float *ptr = ((float*)&kernel.data[(radius+c)*kernel.elemSize()]);
		*ptr = gaussiano;
		sum += *ptr;
	}
	//il ciclo copre tutto il kernel
	for(int c=-radius; c <= radius; c++)
	{
		float *ptr = ((float*)&kernel.data[(radius+c)*kernel.elemSize()]);
		*ptr =  (*ptr/sum);
		//per debug
		//std::cout << "ptr finale:" << *ptr << '\n';
	}
}

//funzione che crea una matrice kernel trasposta
void Transpose(cv::Mat& kernel, cv::Mat& out)
{
	out = cv::Mat(kernel.cols,1,CV_32FC1);

	for(int i = 0; i < kernel.rows; i++)
	{
  	for(int j = 0; j < kernel.cols; j++)
		{
  		*((float *)&out.data[(i + j*out.cols)*out.elemSize()]) = *((float *)&kernel.data[(j + i*kernel.cols)*kernel.elemSize()]);
  	}
	}
	//per debug
	//std::cout << out << '\n';
}

//passo alla funzione i due kernel monodimensionali(orizzontale e verticale), le due matrici di blur orizzontale e Verticale
//stride fisso ad 1 e la matrice finale bidimensionale
void bidimensionalGaussianBlur(cv::Mat& image, cv::Mat& kernelO, cv::Mat& kernelV, cv::Mat& out_bidimensionalBlur, cv::Mat& out_Oblur, cv::Mat& out_Vblur, int stride=1)
{
	//gaussian blur orizzontale
	conv(image,kernelO, out_Oblur, stride);

	//gaussian blur Verticale
	conv(image, kernelV, out_Vblur, stride);

	//gaussian blur Bidimensionale effettuato passando come primo parametro l'immagine di uscita dalla convoluzione orizzontale,
	//come secondo parametro passo il kernel verticale, in questo modo ottengo il blur gaussiano completo
	conv(out_Oblur, kernelV, out_bidimensionalBlur, stride);
}

void sobel(const cv::Mat& image, cv::Mat& magnitude, cv::Mat& orientation)
{
	//dichiaro i kernel convoluzionali orizzontale e verticale
	cv::Mat G_x = (cv::Mat_<float>(3,3) << 1, 0, -1, 2, 0, -2,  1,  0, -1);
	cv::Mat G_y = (cv::Mat_<float>(3,3) << 1, 2,  1, 0, 0,  0, -1, -2, -1);
	//le matrici delle derivate che serviranno nella convFloat
	cv::Mat derivate_x = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);
	cv::Mat derivate_y = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);
	//trovo le derivate applicando la convoluzione float passando come kernel ripsettivamente G_x e G_y per la derivata x e la derivata y
	convFloat(image, G_x, derivate_x);
	convFloat(image, G_y, derivate_y);

	for(int i=0; i < image.rows; i++)
	{
		for(int j=0; j < image.cols; j++)
		{
			float d_x, d_y, arctg;
			d_y = *((float*)&derivate_x.data[(j+i*derivate_x.cols)*derivate_x.elemSize()]);
			d_x = *((float*)&derivate_y.data[(j+i*derivate_y.cols)*derivate_y.elemSize()]);
			//riempio la magnitudo secondo la formula
			*((float*)&magnitude.data[(j+i*image.cols)*magnitude.elemSize()]) = sqrt(pow(d_x,2)+pow(d_y,2));
			//riempio l'orientation secondo la formula
			arctg = atan2(d_y,d_x);
			if(arctg < 0) arctg = arctg + 2*M_PI;
			*((float*)&orientation.data[(j+i*orientation.cols)*orientation.elemSize()]) = arctg;
		}
	}
}

float bilinear(const cv::Mat& image, float r, float c)
{

	float bilinearInterpolation;

	//ci interessa la parte intera quindi utilizzo la funzione floor per estrarla
	int x = floor(r);
	int y = floor(c);
	//differenza tra il punto a coordinate intere e il punto dato.
	float t = r - x;
	float s = c - y;

	// Ottengo il valore in float dei 4 pixel nell'intorno del punto.
	float f00 = *((float*)&image.data[(x * image.cols + y)*image.elemSize()]);
	float f10 = *((float*)&image.data[((x + 1) * image.cols + y)*image.elemSize()]);
	float f01 = *((float*)&image.data[(x * image.cols + (y + 1))*image.elemSize()]);
	float f11 = *((float*)&image.data[((x + 1) * image.cols + (y + 1))*image.elemSize()]);

	bilinearInterpolation = (1 - s)*(1 - t)*f00 + s*(1 - t)*f10 + (1 - s)*t*f01 + s*t*f11;

	return bilinearInterpolation;
}

void findPeaks(const cv::Mat& magnitude, const cv::Mat& orientation, cv::Mat& out ,float th)
{
	float e1, e2;
	float e1x, e1y, e2x,e2y;

	for(int r = 0; r < magnitude.rows; r++)
	{
		for(int c = 0; c < magnitude.cols; c++)
		{
				float *teta = ((float*)&orientation.data[(r*orientation.cols + c)*orientation.elemSize()]);
				//e1
				e1x = (c+1) * cos(*teta);
				e1y = (r+1) * sin(*teta);
				e1 = bilinear(magnitude, e1y, e1x);

				//e2
				e2x = (c-1) * cos(*teta);
				e2y = (r-1) * sin(*teta);
				e2 = bilinear(magnitude, e2y, e2x);

				float *ptr_m = ((float*)&magnitude.data[(r*magnitude.cols + c)*magnitude.elemSize()]);
				//assegno il valore alla matrice di out
				if( *ptr_m >= e1 && *ptr_m >= e2 && *ptr_m >= th)
				{
					*((float*)&out.data[(r*out.cols +c)*out.elemSize()]) = *ptr_m;
				}
				else
				{
					*((float*)&out.data[(r*out.cols +c)*out.elemSize()]) = 0.0;
				}
		}
	}
}


void doubleTh(const cv::Mat& magnitude,cv::Mat& out, float th1, float th2)
{
	for(int r = 0; r < magnitude.rows; r++)
	{
		for(int c = 0; c < magnitude.cols; c++)
		{

				float *ptr_m = (float*)&magnitude.data[(r*magnitude.cols + c)*magnitude.elemSize()];
				//assegno il valore alla matrice di out
				if(*ptr_m > th1 )
				{
					out.at<uint8_t>(r,c) = 255;
				}
				else if( *ptr_m > th2 && *ptr_m <= th1)
				{
					out.at<uint8_t>(r,c) = 128;
				}
				else
				{
					out.at<uint8_t>(r,c) = 0;
				}
		}
	}
}

void canny(const cv::Mat& image, cv::Mat& out, float th, float th1, float th2)
{
	// Sobel.
	cv::Mat orientazione =  cv::Mat::zeros(image.rows,image.cols, CV_32FC1);
  cv::Mat magnitudo =  cv::Mat::zeros(image.rows,image.cols, CV_32FC1);
	sobel(image,magnitudo,orientazione);

	//FindPeaks.
	cv::Mat peaks = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);
	findPeaks(magnitudo, orientazione, peaks, th);

	//isteresi.
	doubleTh(peaks, out, th1, th2);
}


int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;

	std::cout<<"Simple program."<<std::endl;

	//////////////////////
	//parse argument list:
	//////////////////////
	ArgumentList args;
	if(!ParseInputs(args, argc, argv)) {
		return 1;
	}

	while(!exit_loop)
	{
		//generating file name
		//
		//multi frame case
		if(args.image_name.find('%') != std::string::npos)
			sprintf(frame_name,(const char*)(args.image_name.c_str()),frame_number);
		else //single frame case
			sprintf(frame_name,"%s",args.image_name.c_str());

		//opening file
		std::cout<<"Opening "<<frame_name<<std::endl;

		cv::Mat image = cv::imread(frame_name, CV_8UC1);
		if(image.empty())
		{
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}

		//////////////////////

		//chiedo le infomazioni kernel stride da input
		int kernel_size;
		int odd_kernel_size;
		int stride;

		std::cout << "Insert kernel_size:" << '\n';
		std::cin >> kernel_size;
		std::cout << "Insert stride:" << '\n';
		std::cin >> stride;
		std::cout << "Insert odd kernel_size for convolution:" << '\n';
		std::cin >> odd_kernel_size;
		//controllo che il kernel sia dispari
		while(odd_kernel_size % 2 ==0)
		{
			std::cout << "Insert ODD kernel_size" << '\n';
			std::cin >> odd_kernel_size;
		}

		//formula per il calcolo della dimensione dell'immagine finale
		int out_cols = floor(((image.cols - kernel_size)/stride) + 1);
		int out_rows = floor(((image.rows - kernel_size)/stride) + 1);

		//maxpooling
		cv::Mat out_maxPooling(out_rows,out_cols,image.type());
		maxPooling(image, kernel_size, stride, out_maxPooling);
		//averagePooling
		cv::Mat out_averagePooling(out_rows,out_cols,image.type());
		averagePooling(image, kernel_size, stride, out_averagePooling);

		//creo la matrice finale di convoluzione
		cv::Mat out_fconv(image.rows, image.cols, CV_32FC1);
		//matrice kernel float32
		cv::Mat kernel(odd_kernel_size, odd_kernel_size, CV_32FC1);
		//riempimento del kernel con un valore fisso
		for(int i=0; i < kernel.rows; i++)
		{
			for(int j=0; j < kernel.cols;j++)
			{
					*((float*)&kernel.data[(i*kernel.cols + j) * kernel.elemSize()]) = 2.7;
			}
		}
		convFloat(image, kernel, out_fconv);

		//convoluzione intera
		cv::Mat out_Intconv(image.rows,image.cols,CV_8UC1);
		conv(image, kernel,out_Intconv);

		//GAUSSIANO
		int sigma;
		int radius;
		std::cout << "Insert sigma:" << '\n';
		std::cin >> sigma;
		std::cout << "Insert radius:" << '\n';
		std::cin >> radius;

		//creo la matrice kernel gaussiana orizzontale monodimensionale
		cv::Mat kernelG(1,(2*radius+1),CV_32FC1);
		gaussianKernel(sigma,radius, kernelG);

		//creo la matrice kernel gaussiana verticale richiamando la funzione di trasposizione
		cv::Mat kernelV((2*radius+1),1,CV_32FC1);
		Transpose(kernelG, kernelV);
		//per debug
		//std::cout << kernelG << '\n';
		//std::cout << kernelV << '\n';

		//creo le matrici di Blur
		cv::Mat OrizontalBlur(image.rows,image.cols, CV_8UC1);
		cv::Mat VerticalBlur(image.rows,image.cols, CV_8UC1);
		cv::Mat BidimensionalBlur(image.rows,image.cols, CV_8UC1);
		//chiamo la funzione di gaussianBlurBidimensionale
		bidimensionalGaussianBlur(image, kernelG, kernelV, BidimensionalBlur, OrizontalBlur, VerticalBlur);

		//Sobel
		cv::Mat Magnitude = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);
		cv::Mat norm_Magnitude = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
		cv::Mat Orientation = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);
		sobel(image, Magnitude, Orientation);
		normalizeMatrix(Magnitude,norm_Magnitude);

		//findPeaks
		int th1;
		std::cout << "Insert th1 for findPeaks" << '\n';
		std::cin >> th1;
		cv::Mat out_peaks = cv::Mat::zeros(image.rows,image.cols,CV_32FC1);
		findPeaks(Magnitude, Orientation, out_peaks, th1);
		//matrice per la normalizzazione
		cv::Mat out_intPeaks = cv::Mat::zeros(image.rows,image.cols,CV_8UC1);
		normalizeMatrix(out_peaks,out_intPeaks);

		//Isteresi
		int th2;
		std::cout << "Insert th2" << '\n';
		std::cin >> th2;
		//controllo che la soglia th2 sia più piccola di th1
		while(th2 > th1)
		{
			std::cout << "Insert th1 > th2" << '\n';
			std::cin >> th2;
		}
		cv::Mat out_isteresi = cv::Mat::zeros(image.rows,image.cols,CV_8UC1);
		doubleTh(Magnitude, out_isteresi, th1, th2);

		//canny
		cv::Mat out_canny = cv::Mat::zeros(image.rows,image.cols,CV_8UC1);
		//fisso la soglia a 150 per non chiederla da input
		int th = 10;
		canny(image,out_canny,th,th1,th2);


		/////////////////////DISPLAY DELLE IMMAGINI///////////////////////////

		//display original image
		cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
		cv::imshow("image", image);
		//display maxPooling image
		cv::namedWindow("MAXPOOLING", cv::WINDOW_AUTOSIZE);
		cv::imshow("MAXPOOLING", out_maxPooling);
		//display avg_pooling image
		cv::namedWindow("AVGPOOLING", cv::WINDOW_AUTOSIZE);
		cv::imshow("AVGPOOLING", out_averagePooling);

		//display conv image
		cv::namedWindow("CONVOLUTION", cv::WINDOW_AUTOSIZE);
		cv::imshow("CONVOLUTION", out_Intconv);

		//display blur images
		cv::namedWindow("ORIZONTAL BLUR", cv::WINDOW_AUTOSIZE);
		cv::imshow("ORIZONTAL BLUR",OrizontalBlur);

		cv::namedWindow("VERTICAL BLUR", cv::WINDOW_AUTOSIZE);
		cv::imshow("VERTICAL BLUR", VerticalBlur);

		cv::namedWindow("BIDIMENSIONAL BLUR", cv::WINDOW_AUTOSIZE);
		cv::imshow("BIDIMENSIONAL BLUR", BidimensionalBlur);

		//display sobel image
		cv::namedWindow("SOBEL", cv::WINDOW_AUTOSIZE);
		cv::imshow("SOBEL",norm_Magnitude);

		cv::Mat adjMap;
		cv::convertScaleAbs(Orientation, adjMap, 255 / (2*M_PI));
		cv::Mat falseColorsMap;
		cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_AUTUMN);
		cv::imshow("Orientation", falseColorsMap);

		//diplay findPeaks image
		cv::namedWindow("PEAKS", cv::WINDOW_AUTOSIZE);
		cv::imshow("PEAKS", out_intPeaks);

		//display Isteresi image
		cv::namedWindow("ISTERESI", cv::WINDOW_AUTOSIZE);
		cv::imshow("ISTERESI", out_isteresi);

		//display canny image
		cv::namedWindow("CANNY", cv::WINDOW_AUTOSIZE);
		cv::imshow("CANNY", out_canny);

		//wait for key or timeout
		unsigned char key = cv::waitKey(args.wait_t);
		std::cout<<"key "<<int(key)<<std::endl;

		//here you can implement some looping logic using key value:
		// - pause
		// - stop
		// - step back
		// - step forward
		// - loop on the same frame
		if(key == 'q')
			exit_loop = true;

		frame_number++;
	}

	return 0;
}
