/*
VULTAGGIO SALVO
305012

NOTE:
---- è stata utilizzata la funzione di opencv GaussianBlur perchè con la mia versione
----(bidimensionalGaussianBlur), che sembra essere funzionante, la funzoine Harrys non trova correttamente i keypoints

---- il programma eseguito carica automanticamente dalla cartella images la nuova copertina e la mostra in una finestra apposita
*/

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <iterator>

// eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

//time
#include <time.h>

void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out, int stride=1)
{
	float conv;
	int a = floor((kernel.rows -1)/2);
	int b = floor((kernel.cols -1)/2);

	// controllo che il kernel sia simmetrico (dimensioni dispari)
	if (kernel.rows % 2 != 1 || kernel.cols % 2 != 1)
	{
		std::cerr << "- Kernel non simmetrico -";
		exit(1);
	}

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
	out = cv::Mat(image.rows, image.cols, CV_32FC1);
	cv::Mat processingFloat = cv::Mat(image.rows,image.cols,CV_32FC1);
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
	normalized = cv::Mat(sourceMatrix.rows,sourceMatrix.cols,CV_8UC1);

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
	float sum = 0.0;
	float dim_kernel = (2*radius)+1;
	float normalizedGaussianValue;
	kernel = cv::Mat(1,dim_kernel,CV_32FC1);
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
	}

	// normalizzazione della distribuzione gaussiana
	for(int i = 0; i < dim_kernel; ++i)
	{
		normalizedGaussianValue = *((float *) &kernel.data[i * kernel.elemSize()]) / sum;
		*((float *) &kernel.data[i * kernel.elemSize()]) = normalizedGaussianValue;
	}
}

//passo alla funzione i due kernel monodimensionali(orizzontale e verticale), le due matrici di blur orizzontale e Verticale
//stride fisso ad 1 e la matrice finale bidimensionale
void bidimensionalGaussianBlur(cv::Mat& image, cv::Mat& kernelO, cv::Mat& kernelV, cv::Mat& out_bidimensionalBlur, int stride=1)
{
	cv::Mat out_Oblur;
	conv(image,kernelO, out_Oblur, stride);
	conv(out_Oblur, kernelV, out_bidimensionalBlur, stride);
}

//questa funzione usa il kernel di sobel per calcolare le derivate
void sobel_derivate(const cv::Mat& image,cv::Mat& derivate_x,cv::Mat& derivate_y)
{
	//dichiaro i kernel convoluzionali orizzontale e verticale
	cv::Mat G_x = (cv::Mat_<float>(3,3) << 1, 0, -1, 2, 0, -2,  1,  0, -1);
	cv::Mat G_y = (cv::Mat_<float>(3,3) << 1, 2,  1, 0, 0,  0, -1, -2, -1);

	//trovo le derivate applicando la convoluzione float passando come kernel ripsettivamente G_x e G_y per la derivata x e la derivata y
	convFloat(image, G_x, derivate_x);
	convFloat(image, G_y, derivate_y);
}

void myHarrisCornerDetector(const cv::Mat image, std::vector<cv::KeyPoint> & keypoints0, float alpha, float harrisTh)
{
	//matrici delle derivate
	cv::Mat I_x = cv::Mat(image.rows,image.cols,CV_32FC1);
	cv::Mat I_y = cv::Mat(image.rows,image.cols,CV_32FC1);
	//matrici delle derivate seconde
	cv::Mat I_x2 = cv::Mat(image.rows,image.cols,CV_32FC1);
	cv::Mat I_y2 = cv::Mat(image.rows,image.cols,CV_32FC1);
	//derivata mista
	cv::Mat I_xy = cv::Mat(image.rows,image.cols,CV_32FC1);
	//matrici gaussiane
	cv::Mat gaussianx2 = cv::Mat(image.rows,image.cols,CV_32FC1);
	cv::Mat gaussiany2 = cv::Mat(image.rows,image.cols,CV_32FC1);
	cv::Mat gaussianxy = cv::Mat(image.rows,image.cols,CV_32FC1);

	//faccio le derivate x e y col kernel di sobel
	sobel_derivate(image,I_x,I_y);
	//faccio le derivate seconde al quadrato e miste
	I_xy = I_x.mul(I_y);
	I_x2 = I_x.mul(I_x);
	I_y2 = I_y.mul(I_y);

	//creo i lkernel gaussiano
	cv::Mat G_kernel;
	gaussianKernel(2,1,G_kernel);
	cv::Mat G_kernelt = G_kernel.t();

	//effettuo il bulr gaussiano con la funzioen di opencv
	cv::GaussianBlur(I_x2,gaussianx2, cv::Size( 3, 3 ), 0, 0 );
	cv::GaussianBlur(I_y2,gaussiany2, cv::Size( 3, 3 ), 0, 0 );
	cv::GaussianBlur(I_xy,gaussianxy, cv::Size( 3, 3 ), 0, 0 );
/*
	bidimensionalGaussianBlur(I_x2,G_kernel,G_kernelt,gaussianx2);
	bidimensionalGaussianBlur(I_y2,G_kernel,G_kernelt,gaussiany2);
	bidimensionalGaussianBlur(I_xy,G_kernel,G_kernelt,gaussianxy);
*/
	//valore di theta
	cv::Mat theta = cv::Mat(image.rows,image.cols,CV_32FC1);
	theta = gaussianx2.mul(gaussiany2) - gaussianxy.mul(gaussianxy) - (alpha*((gaussianx2 + gaussiany2).mul(gaussianx2 + gaussiany2)));

	//effettuo la non maximum SUPPRESSION
	for(int i=0;i<theta.rows;i++)
	{
		for(int j=0;j<theta.cols;j++)
		{
			//il primo controllo impone di eliminare tutti i punti che sono <= della soglia, quindi se maggiore della soglia lo inserisco nel vettore di KeyPoint
			if(theta.at<float>(i,j) > harrisTh)
			{
				//il secondo controllo verifica se in un intorno 3x3 c'è un massimo locale
				//float maxloc = theta.at<float>(i,j);
				for(int a=-1;a<=1;a++)
				{
					if(theta.at<float>(i,j) < theta.at<float>(i+a,j+a))
					{
						//inserisco nel vettore di KeyPoint il massimo locale
						keypoints0.push_back( cv::KeyPoint(float(j+a), float(i+a), 3.f) );
					}
				}
			}
		}
	}

	/*
	// Per la response di Harris:
      cv::Mat adjMap;
	    cv::Mat falseColorsMap;
	    double minr,maxr;

	    cv::minMaxLoc(theta, &minr, &maxr);
	    cv::convertScaleAbs(theta, adjMap, 255 / (maxr-minr));
	    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
	    cv::namedWindow("response1", cv::WINDOW_NORMAL);
	    cv::imshow("response1", falseColorsMap);
		*/
}

void myFindHomographySVD(const std::vector<cv::Point2f> & points1, const std::vector<cv::Point2f> & points0, cv::Mat & H)
{
	cv::Mat A(points1.size()*2,9, CV_64FC1, cv::Scalar(0));
	cv::Mat D;
	cv::Mat U;
	cv::Mat Vt;

	//costrusco A col for dividendo in indice pari e dispari
	//gli elementi non indicati sono per costruzione di A già posti a 0
	for(int i=0; i<A.rows;i+=2)
	{
		A.at<double>(i,0) = - points1[i/2].x;
		A.at<double>(i,1) = - points1[i/2].y;
		A.at<double>(i,2) = -1;
		A.at<double>(i,6) = points0[i/2].x  * points1[i/2].x;
		A.at<double>(i,7) = points0[i/2].x  * points1[i/2].y;
		A.at<double>(i,8) = points0[i/2].x;
	}
	for(int i=1; i<A.rows;i+=2)
	{
		A.at<double>(i,3) = - points1[i/2].x;
		A.at<double>(i,4) = - points1[i/2].y;
		A.at<double>(i,5) = -1;
		A.at<double>(i,6) = points0[i/2].y  * points1[i/2].x;
		A.at<double>(i,7) = points0[i/2].y  * points1[i/2].y;
		A.at<double>(i,8) = points0[i/2].y;
	}

	//std::cout << "matrice A" <<A<< '\n';
	//std::cout << "*********************" << '\n';
	cv::SVD::compute(A, D, U, Vt,cv::SVD::FULL_UV);
	cv::Mat V = Vt.t();

	//debug
	/*
	std::cout << "Vt:" <<Vt<< '\n';
	std::cout << "**********************" << '\n';
	std::cout << "V:" <<V<< '\n';
	*/


	int count = 0;
	H = cv::Mat(3,3,CV_64FC1);
	//h è l'ultima colonna di V
	for(int i=0;i<H.rows;i++)
	{
		for(int j=0;j<H.cols;j++)
		{
			H.at<double>(i,j) = V.at<double>(count,8);
			count++;
		}
	}

	H/=H.at<double>(2,2);

	//debug
	//std::cout<<"myH"<<std::endl<<H<<std::endl;

	A.release();
	V.release();
	D.release();
	U.release();
}

void myFindHomographyRansac(const std::vector<cv::Point2f> & points1, const std::vector<cv::Point2f> & points0, const std::vector<cv::DMatch> & matches, int N,
														float epsilon, int sample_size, cv::Mat & H, std::vector<cv::DMatch> & matchesInlierBest)
{
	std::vector<cv::Point2f> sample_0 , sample_1;
	int randomValue[points1.size()];
	std::vector<cv::Point2f> inlier_0, inlier_1;
	std::vector<cv::Point2f> inliers_best0,inliers_best1;

	srand(time(NULL));

	//inizio del ciclo RANSAC
	for(int r=0; r<N; r++)
	{
		//std::cout << " ciclo:" <<r;

		//seleziono 4 match a caso da mettere nel vettore
		for (int i = 0; i < sample_size; i++)
		{
			randomValue[i] = rand() % points0.size();
		}

		//li metto nelle immagini sample
		for (int i = 0; i < sample_size ; i++)
		{
			sample_0.push_back(points0[randomValue[i]]) ;
			sample_1.push_back(points1[randomValue[i]]) ;
		}

		//opencv
		//H = cv::findHomography( cv::Mat(sample_1), cv::Mat(sample_0), 0);

		cv::Mat Hv;
		myFindHomographySVD(sample_1,sample_0, Hv);

		//std::cout << H << std::endl << Hv << std::endl;

		for(unsigned int i=0; i < points0.size(); i++)
		{
			cv::Mat newPoint1 = cv::Mat(3,1,CV_64FC1);
			cv::Mat newPoint0;

			newPoint1 = cv::Mat(cv::Point3d(points1[i].x, points1[i].y,1));
			newPoint0 = cv::Mat(cv::Point2d(points0[i].x,points0[i].y));

			newPoint1.convertTo(newPoint1,CV_64FC1);

			cv::Mat h;
			h = Hv * newPoint1;
			h /= h.at<double>(2,0);
			cv::Mat h_2 = cv::Mat(cv::Point2d(h.at<double>(0,0), h.at<double>(1,0)));

			//effettuo il controllo della norma
			if(cv::norm(newPoint0,h_2) < epsilon)
			{
				inlier_0.push_back(points0[i]);
				inlier_1.push_back(points1[i]);
			}
		}

		if(inlier_0.size() > inliers_best0.size() )
		{
			//svuoto gli inlier_best0 e inlier_best1
			inliers_best0.clear();
			inliers_best1.clear();

			for(int j=0; j< (int) inlier_0.size(); j++)
			{
				inliers_best0.push_back(inlier_0[j]);
				inliers_best1.push_back(inlier_1[j]);
			}
		}

		Hv.release();
		inlier_1.clear();
		inlier_0.clear();
		sample_0.clear();
		sample_1.clear();
	}

	//H = cv::findHomography( cv::Mat(inliers_best1), cv::Mat(inliers_best0), 0);

	myFindHomographySVD(inliers_best1, inliers_best0, H);

	for(unsigned int i=0;i<inliers_best0.size();i++)
	{
		for(unsigned int j=0;j<points0.size();j++)
		{
			if(inliers_best1[i] == points1[j] && inliers_best0[i] == points0[j])
			{
				matchesInlierBest.push_back(matches[j]);
			}
		}
	}
}

// Funzione per la sovrapposizione copertina
void replaceCover(cv::Mat& toReplace, cv::Mat& replace, cv::Mat& out)
{
	out = cv::Mat(toReplace.rows, toReplace.cols, CV_8UC1);

	for(int r = 0; r < toReplace.rows; r++)
		for(int c = 0; c < toReplace.cols; c++)
		{
			if(replace.at<u_int8_t>(r,c) != 0)
					out.at<u_int8_t>(r,c) = replace.at<u_int8_t>(r,c);
			else
				out.at<u_int8_t>(r,c) = toReplace.at<u_int8_t>(r,c);
		}
}


int main(int argc, char **argv) {

    if (argc < 3)
    {
        std::cerr << "Usage prova <image_filename> <book_filename>" << std::endl;
        return 0;
    }

    // images
    cv::Mat input, cover, newCover;

    // load image from file
    input = cv::imread(argv[1], CV_8UC1);
	if(input.empty())
	{
		std::cout<<"Error loading input image "<<argv[1]<<std::endl;
		return 1;
	}

    // load image from file
    cover = cv::imread(argv[2], CV_8UC1);
	if(cover.empty())
	{
		std::cout<<"Error loading book image "<<argv[2]<<std::endl;
		return 1;
	}

	// load newCover
		newCover = cv::imread("images/no.jpg", CV_8UC1);
	if(newCover.empty())
	{
		std::cout<<"Error loading newCover image "<<std::endl;
		return 1;
	}

//funzione che permette di fare il resize dell'immagine in modo da scegliere qualsisi tipo di copertina
	if(newCover.rows != 431 || newCover.cols != 574)
	{
		cv::Mat dst;
		cv::resize(newCover, dst, cv::Size(431,574));

		newCover = dst;
	}


	////////////////////////////////////////////////////////
	/// HARRIS CORNER
	//
	float alpha = 0.04;
	float harrisTh =  999900000;    //da impostare in base alla propria implementazione!!!!!

	std::vector<cv::KeyPoint> keypoints0, keypoints1;

	myHarrisCornerDetector(input, keypoints0, alpha, harrisTh);
	myHarrisCornerDetector(cover, keypoints1, alpha, harrisTh);

/*
	{
		std::vector<cv::Point2f> corners;
		int maxCorners = 0;
		double qualityLevel = 0.01;
		double minDistance = 10;
		int blockSize = 3;
		bool useHarrisDetector = true;
		double k = 0.04;

		cv::goodFeaturesToTrack( input,corners,maxCorners,qualityLevel,minDistance,cv::noArray(),blockSize,useHarrisDetector,k );
		std::transform(corners.begin(), corners.end(), std::back_inserter(keypoints0), [](const cv::Point2f & p){ return cv::KeyPoint(p.x,p.y,3.0);} );

		corners.clear();
		cv::goodFeaturesToTrack( cover, corners, maxCorners, qualityLevel, minDistance, cv::noArray(), blockSize, useHarrisDetector, k );
		std::transform(corners.begin(), corners.end(), std::back_inserter(keypoints1), [](const cv::Point2f & p){ return cv::KeyPoint(p.x,p.y,3.0);} );
	}
	*/

	std::cout<<"keypoints0 "<<keypoints0.size()<<std::endl;
	std::cout<<"keypoints1 "<<keypoints1.size()<<std::endl;



	////////////////////////////////////////////////////////
  /// CALCOLO DESCRITTORI E MATCHES
	//
    int briThreshl=30;
    int briOctaves = 3;
    int briPatternScales = 1.0;
		cv::Mat descriptors0, descriptors1;

	//dichiariamo un estrattore di features di tipo BRISK
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::BRISK::create(briThreshl, briOctaves, briPatternScales);
    //calcoliamo il descrittore di ogni keypoint
    extractor->compute(input, keypoints0, descriptors0);
    extractor->compute(cover, keypoints1, descriptors1);

    //associamo i descrittori tra me due immagini
    std::vector<std::vector<cv::DMatch> > matches;
    std::vector<cv::DMatch> matchesDraw;
		cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING);
		//matcher.radiusMatch(descriptors0, descriptors1, matches, input.cols*2.0);
		matcher.match(descriptors0, descriptors1, matchesDraw);

    //copio i match dentro a dei semplici vettori oint2f
    std::vector<cv::Point2f> points[2];
    for(unsigned int i=0; i<matchesDraw.size(); ++i)
      {
    	points[0].push_back(keypoints0.at(matchesDraw.at(i).queryIdx).pt);
    	points[1].push_back(keypoints1.at(matchesDraw.at(i).trainIdx).pt);
      }
	////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////
    // CALCOLO OMOGRAFIA
    //
    //
    cv::Mat H;                                  //omografia finale
    std::vector<cv::DMatch> matchesInliersBest; //match corrispondenti agli inliers trovati
    std::vector<cv::Point2f> corners_cover;     //coordinate dei vertici della cover sull'immagine di input
    bool have_match=false;                      //verra' messo a true in caso ti match

	//
	// Verifichiamo di avere almeno 4 inlier per costruire l'omografia
	//
	//
    if(points[0].size()>=4)
    {
    	//
    	// Soglie RANSAC
    	//
    	// Piuttosto critiche, da adattare in base alla propria implementazione
    	//
    	int N=2000;            //numero di iterazioni di RANSAC
    	float epsilon = 10;      //distanza per il calcolo degli inliers


    	// Dimensione del sample per RANSAC, quiesto e' fissato
    	//
    	int sample_size = 4;    //dimensione del sample di RANSAC

    	//////////
    	// FASE 2
    	//
    	//
    	//
    	// Inizialmente utilizzare questa chiamata OpenCV, che utilizza RANSAC, per verificare i vostri corner di Harris
    	//
    	//
			/*
    	cv::Mat mask;
    	H = cv::findHomography( cv::Mat(points[1]), cv::Mat(points[0]), CV_RANSAC, 3, mask);
    	for(int i=0;i<matchesDraw.size();++i)
    		if(mask.at<uchar>(0,i) == 1) matchesInliersBest.push_back(matchesDraw[i]);
				*/
    	//
    	//
    	//
    	// Una volta che i vostri corner di Harris sono funzionanti, commentare il blocco sopra e abilitare la vostra myFindHomographyRansac
    	//
    	myFindHomographyRansac(points[1], points[0], matchesDraw, N, epsilon, sample_size, H, matchesInliersBest);
    	//
    	//
    	//

    	std::cout<<std::endl<<"Risultati Ransac: "<<std::endl;
    	std::cout<<"Num inliers / match totali  "<<matchesInliersBest.size()<<" / "<<matchesDraw.size()<<std::endl;

      std::cout<<"H"<<std::endl<<H<<std::endl;

    	//
    	// Facciamo un minimo di controllo sul numero di inlier trovati
    	//
    	//
    	float match_kpoints_H_th = 0.1;
    	if(matchesInliersBest.size() > matchesDraw.size()*match_kpoints_H_th)
    	{
    		std::cout<<"MATCH!"<<std::endl;
    		have_match = true;


    		// Calcoliamo i bordi della cover nell'immagine di input, partendo dai corrispondenti nell'immagine target
    		//
    		//
    		cv::Mat p  = (cv::Mat_<double>(3, 1) << 0, 0, 1);
    		cv::Mat pp = H*p;
    		pp/=pp.at<double>(2,0);
    		std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
    		if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
    		{
    			corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
    		}

    		p  = (cv::Mat_<double>(3, 1) << cover.cols-1, 0, 1);
    		pp = H*p;
    		pp/=pp.at<double>(2,0);
    		std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
    		if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
    		{
    			corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
    		}

    		p  = (cv::Mat_<double>(3, 1) << cover.cols-1, cover.rows-1, 1);
    		pp = H*p;
    		pp/=pp.at<double>(2,0);
    		std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
    		if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
    		{
    			corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
    		}

    		p  = (cv::Mat_<double>(3, 1) << 0,cover.rows-1, 1);
    		pp = H*p;
    		pp/=pp.at<double>(2,0);
    		std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
    		if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
    		{
    			corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
    		}
    	}
    	else
    	{
    		std::cout<<"Pochi inliers! "<<matchesInliersBest.size()<<"/"<<matchesDraw.size()<<std::endl;
    	}


    }
    else
    {
    	std::cout<<"Pochi match! "<<points[0].size()<<"/"<<keypoints0.size()<<std::endl;
    }
	////////////////////////////////////////////////////////

    ////////////////////////////////////////////
    /// WINDOWS
    cv::Mat inputKeypoints;
    cv::Mat coverKeypoints;
    cv::Mat outMatches;
    cv::Mat outInliers;

    cv::drawKeypoints(input, keypoints0, inputKeypoints);
    cv::drawKeypoints(cover, keypoints1, coverKeypoints);

    cv::drawMatches(input, keypoints0, cover, keypoints1, matchesDraw, outMatches);
    cv::drawMatches(input, keypoints0, cover, keypoints1, matchesInliersBest, outInliers);


    // se abbiamo un match, disegniamo sull'immagine di input i contorni della cover
    if(have_match)
    {
    	for(unsigned int i = 0;i<corners_cover.size();++i)
    	{
    		cv::line(input, cv::Point(corners_cover[i].x , corners_cover[i].y ), cv::Point(corners_cover[(i+1)%corners_cover.size()].x , corners_cover[(i+1)%corners_cover.size()].y ), cv::Scalar(255), 2, 8, 0);
    	}
    }

		// Sostituzione cover con una a piacere
		///////////////////////////////////////////////
		cv::Mat warpedCover;
		cv::warpPerspective(newCover, warpedCover, H, input.size());

		cv::namedWindow("New Book Cover", cv::WINDOW_AUTOSIZE);
		cv::imshow("New Book Cover", newCover);

		cv::Mat out;
		replaceCover(input, warpedCover, out);

		cv::namedWindow("Swapped cover", cv::WINDOW_AUTOSIZE);
		cv::imshow("Swapped cover", out);
		//////////////////////////////
		/////////////////////////////

    cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input", input);

    cv::namedWindow("BookCover", cv::WINDOW_AUTOSIZE);
    cv::imshow("BookCover", cover);

    cv::namedWindow("inputKeypoints", cv::WINDOW_AUTOSIZE);
    cv::imshow("inputKeypoints", inputKeypoints);

    cv::namedWindow("coverKeypoints", cv::WINDOW_AUTOSIZE);
    cv::imshow("coverKeypoints", coverKeypoints);

    cv::namedWindow("Matches", cv::WINDOW_AUTOSIZE);
    cv::imshow("Matches", outMatches);

    cv::namedWindow("Matches Inliers", cv::WINDOW_AUTOSIZE);
    cv::imshow("Matches Inliers", outInliers);

    cv::waitKey();

    return 0;
}
