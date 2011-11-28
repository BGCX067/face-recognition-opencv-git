/**
*
*   @Brief: Clase que implementa la funcionalidad de reconocimiento de rostros
*
*
**/

#include <fstream>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include "cv.h"
#include "cvaux.h"
#include "highgui.h"

using namespace std;


/**
*   @TODO: Actualmente todo el entorno es publico
*
**/
class CFace
{

public:

    //      MEMBERS       =======================
    IplImage ** faceImgArr        ; // array of face images
    CvMat    *  personNumTruthMat ; // array of person numbers
    int nTrainFaces               ; // the number of training images
    int nEigens                   ; // the number of eigenvalues
    IplImage * pAvgTrainImg       ; // the average image
    IplImage ** eigenVectArr      ; // eigenvectors
    CvMat * eigenValMat           ; // eigenvalues
    CvMat * projectedTrainFaceMat ; // projected training faces
    CvHaarClassifierCascade *cascade_f;
    CvMemStorage			*storage; // Almacenamiento de informacion
    int temporal                  ;  // variable para controlar aprendizaje o reiniciarlo

    //      METHODS       ========================
    CFace();
    ~CFace();

    void    Execute_Capture(void);
    void    learn(void);
    int*    recognize(void);
    void    doPCA(void);
    void    storeTrainingData(void);
    int     loadTrainingData(CvMat** pTrainPersonNumMat);
    int     findNearestNeighbor(float* projectedTestFace);
    int     loadFaceImgArray(char* filename);
    void    detectFaces(IplImage* img, char c);
};

/** @brief loadTrainingData
  *
  */
int CFace::loadTrainingData(CvMat** pTrainPersonNumMat)
{
    CvFileStorage * fileStorage;
    int i;

    // create a file-storage interface
    fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_READ );
    if( !fileStorage )
    {
        fprintf(stderr, "Can't open facedata.xml\n");
        return 0;
    }

    nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
    nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
    *pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
    eigenValMat  = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
    projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
    pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
    eigenVectArr = (IplImage **)cvAlloc(nTrainFaces*sizeof(IplImage *));
    for(i=0; i<nEigens; i++)
    {
        char varname[200];
        sprintf( varname, "eigenVect_%d", i );
        eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
    }

    // release the file-storage interface
    cvReleaseFileStorage( &fileStorage );

    return 1;

}

/** @brief loadFaceImgArray
  *
  */
int CFace::loadFaceImgArray(char* filename)
{
        FILE * imgListFile = 0;
    char imgFilename[512];
    int iFace, nFaces=0;


    // open the input file
    if( !(imgListFile = fopen(filename, "r")) )
    {
        fprintf(stderr, "Can\'t open file %s\n", filename);
        return 0;
    }

    // count the number of faces
    while( fgets(imgFilename, 512, imgListFile) ) ++nFaces;
    rewind(imgListFile);

    // allocate the face-image array and person number matrix
    faceImgArr        = (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
    personNumTruthMat = cvCreateMat( 1, nFaces, CV_32SC1 );

    // store the face images in an array
    for(iFace=0; iFace<nFaces; iFace++)
    {
        // read person number and name of image file
        fscanf(imgListFile,
               "%d %s", personNumTruthMat->data.i+iFace, imgFilename);

        // load the face image
        faceImgArr[iFace] = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);

        if( !faceImgArr[iFace] )
        {
            fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
            return 0;
        }
    }

    fclose(imgListFile);

    return nFaces;
}

/** @brief findNearestNeighbor
  *
  */
int CFace::findNearestNeighbor(float* projectedTestFace)
{
    double leastDistSq = DBL_MAX;
    int i, iTrain, iNearest = 0;

    for(iTrain=0; iTrain<nTrainFaces; iTrain++)
    {
        double distSq=0;

        for(i=0; i<nEigens; i++)
        {
            float d_i = projectedTestFace[i] - projectedTrainFaceMat->data.fl[iTrain*nEigens + i];
            distSq += d_i*d_i; // Euclidean
        }

        if(distSq < leastDistSq)
        {
            leastDistSq = distSq;
            iNearest = iTrain;
        }
        if(100000000 < leastDistSq)
        {
            iNearest = -1;
        }
    }
    return iNearest;
}

/** @brief storeTrainingData
  *
  */
void CFace::storeTrainingData(void)
{
    CvFileStorage * fileStorage;
    int i;

    // create a file-storage interface
    fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_WRITE );

    // store all the data
    cvWriteInt( fileStorage, "nEigens", nEigens );
    cvWriteInt( fileStorage, "nTrainFaces", nTrainFaces );
    cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0,0));
    cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0,0));
    cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0,0));
    cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));
    for(i=0; i<nEigens; i++)
    {
        char varname[200];
        sprintf( varname, "eigenVect_%d", i );
        cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
    }

    // release the file-storage interface
    cvReleaseFileStorage( &fileStorage );

}

/** @brief doPCA
  *
  */
void CFace::doPCA(void)
{
        int i;
    CvTermCriteria calcLimit;
    CvSize faceImgSize;

    // set the number of eigenvalues to use
    nEigens = nTrainFaces-1;

    // allocate the eigenvector images
    faceImgSize.width  = faceImgArr[0]->width;
    faceImgSize.height = faceImgArr[0]->height;
    eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
    for(i=0; i<nEigens; i++)
        eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

    // allocate the eigenvalue array
    eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

    // allocate the averaged image
    pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

    // set the PCA termination criterion
    calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);

    // compute average image, eigenvalues, and eigenvectors
    cvCalcEigenObjects(
        nTrainFaces,
        (void*)faceImgArr,
        (void*)eigenVectArr,
        CV_EIGOBJ_NO_CALLBACK,
        0,
        0,
        &calcLimit,
        pAvgTrainImg,
        eigenValMat->data.fl);

    cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);

}

/** @brief recognize
  *
  */
int* CFace::recognize(void)
{
        int iNearest, nearest, truth;
    int i, nTestFaces  = 0;         // the number of test images
    CvMat * trainPersonNumMat = 0;  // the person numbers during training
    float * projectedTestFace = 0;

    // load test images and ground truth for person number
    nTestFaces = loadFaceImgArray("test.txt");
    printf("%d test faces loaded\n", nTestFaces);

    // load the saved training data
    if( !loadTrainingData( &trainPersonNumMat ) ) return 0;

    // project the test images onto the PCA subspace
    projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );

    int* respNearest = new int[nTestFaces];

    for(i=0; i<nTestFaces; i++)
    {
        // project the test image onto the PCA subspace
        cvEigenDecomposite(
            faceImgArr[i],
            nEigens,
            eigenVectArr,
            0, 0,
            pAvgTrainImg,
            projectedTestFace);

        iNearest = findNearestNeighbor(projectedTestFace);
        truth    = personNumTruthMat->data.i[i];
        if(iNearest!= -1) nearest  = trainPersonNumMat->data.i[iNearest];
        else nearest = -1;
        printf("nearest = %d, Truth = %d\n", nearest, truth);
        respNearest[i] =  nearest;
    }
    return respNearest;
}

/** @brief learn
  *
  */
void CFace::learn(void)
{
     int i, offset;

    // load training data
    nTrainFaces = loadFaceImgArray("train.txt");
    if( nTrainFaces < 2 )
    {
        fprintf(stderr,
                "Need 2 or more training faces\n"
                "Input file contains only %d\n", nTrainFaces);
        return;
    }

    // do PCA on the training faces
    doPCA();

    // project the training images onto the PCA subspace
    projectedTrainFaceMat = cvCreateMat( nTrainFaces, nEigens, CV_32FC1 );
    offset = projectedTrainFaceMat->step / sizeof(float);
    for(i=0; i<nTrainFaces; i++)
    {
        //int offset = i * nEigens;
        cvEigenDecomposite(
            faceImgArr[i],
            nEigens,
            eigenVectArr,
            0, 0,
            pAvgTrainImg,
            //projectedTrainFaceMat->data.fl + i*nEigens);
            projectedTrainFaceMat->data.fl + i*offset);
    }

    // store the recognition data as an xml file
    storeTrainingData();

}

/** @brief Constructor de la clase CFace
  *
  */
CFace::CFace()
{
    // Inicializando variables
    faceImgArr            = 0; // array of face images
    personNumTruthMat     = 0; // array of person numbers
    nTrainFaces           = 0; // the number of training images
    nEigens               = 0; // the number of eigenvalues
    pAvgTrainImg          = 0; // the average image
    eigenVectArr          = 0; // eigenvectors
    eigenValMat           = 0; // eigenvalues
    projectedTrainFaceMat = 0; // projected training faces
    temporal              = 0; // variable para controlar aprendizaje o reiniciarlo

    char *file1 = "haarcascade_frontalface_alt.xml";
    cascade_f = (CvHaarClassifierCascade*)cvLoad(file1, 0, 0, 0);
    storage = cvCreateMemStorage(0);
}

/** @brief Se encarga de iniciar la captura por camara
  *
  */
void CFace::Execute_Capture(void)
{
    CvCapture* capture = cvCreateCameraCapture(0);
    IplImage* frame;
    cvNamedWindow( "Video", CV_WINDOW_AUTOSIZE );
    while(1)
    {
        char key;
        frame = cvQueryFrame( capture );

        if( !frame )                    // Si no logro capturar nada, break
            break;

        key = cvWaitKey(33);
        detectFaces(frame, key);        // LLamo a la funcion detectFaces
        cvShowImage("Video", frame);    // Muestro la captura en un frame
        if( key == 27 )                 // en caso se presione la tecla ESC
            break;
    }
    cvReleaseCapture( &capture );
    cvDestroyWindow( "Video" );
}

/** @brief Realiza la deteccion de un rostro dentro de un frame
  *
  */
void CFace::detectFaces(IplImage* img, char key)
{
    int i,j;
    CvSeq *faces = cvHaarDetectObjects(img, cascade_f, storage,	1.1, 5, 0, cvSize(130, 130));

    if (faces->total == 0)
        return;

    CvRect *r = (CvRect*)cvGetSeqElem(faces, 0);
    cvRectangle(img, cvPoint(r->x, r->y),
                cvPoint(r->x + r->width, r->y + r->height),
                CV_RGB(255, 0, 0), 1, 8, 0);

    int channels  = img->nChannels;
    IplImage* img1 = cvCreateImage(cvSize(r->height, r->width), img->depth, channels);
    int a, b, k;

    for (i = r->y, a = 0; i < r->y + r->height, a < r->height; ++i, ++a)
    {
        for (j = r->x, b = 0; j < r->x + r->width, b < r->width; ++j, ++b)
        {
            for ( k = 0; k < channels; ++k)
            {
                CvScalar tempo = cvGet2D(img, i, j);
                img1->imageData[a * img1->widthStep + b * channels + k] =
                    (char)tempo.val[k];
            }
        }
    }

    fstream f;
    char cadena[100];
    int nper, *resp;
    if (key == 's')
    {
        CvSize size = cvSize(100,100);
        IplImage* tmpsize = cvCreateImage(size,img->depth, channels);
        cvResize(img1,tmpsize,CV_INTER_LINEAR);
        f.open("nper.txt", fstream::in);
        f >> nper;
        f.close();
        nper++;
        f.open("nper.txt", fstream::out);
        f << nper;
        f.close();
        f.open("train.txt", fstream::app);
        cin >> cadena;
        f << nper << " " << cadena << "\n";
        f.close();
        cvSaveImage(cadena, tmpsize);
        learn();
    }
    if (key == 'r')
    {
        temporal = 1;
    }
    if (key == 'e')
    {
        temporal = 0;
    }
    if (temporal)
    {
        f.open("test.txt", fstream::out);
        for (int nf = 0; nf < (faces ? faces->total : 0); ++nf)
        {
            CvRect *r = (CvRect*)cvGetSeqElem(faces, nf);
            cvRectangle(img, cvPoint(r->x, r->y),
                        cvPoint(r->x + r->width, r->y + r->height),
                        CV_RGB(255, 0, 0), 1, 8, 0);
            int channels  = img->nChannels;
            IplImage* img1 =
                cvCreateImage(cvSize(r->height, r->width), img->depth, channels);
            int a, b, k;
            for(i=r->y, a=0; i<r->y+r->height, a<r->height; ++i, ++a)
            {
                for(j=r->x, b=0; j<r->x+r->width, b<r->width; ++j, ++b)
                {
                    for(k=0; k<channels; k++)
                    {
                        CvScalar tempo = cvGet2D(img, i, j);
                        img1->imageData[a * img1->widthStep + b * channels + k] =
                            (char)tempo.val[k];
                    }
                }
            }
            CvSize size = cvSize(100, 100);
            IplImage* tmpsize = cvCreateImage(size, img->depth, channels);
            cvResize(img1, tmpsize, CV_INTER_LINEAR);
            char varname[200];
            sprintf(varname, "consulta_%d.bmp", nf );
            f << 1 << " " << varname << " \n";
            cvSaveImage(varname, tmpsize);
        }
        f.close();
        resp = recognize();
        for(j = 0; j < faces->total; ++j)
        {
            CvRect *r = (CvRect*)cvGetSeqElem(faces, j);
            f.open("train.txt", fstream::in);
            sprintf(cadena, "Desconocido");
            for (i = 0; i < resp[j]; ++i)
            {
                f.getline(cadena,100);
            }
            CvFont font;
            cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);
            cvPutText(img, cadena, cvPoint(r->x, r->y+r->height/2), &font,
                      cvScalar(255, 255, 255, 0));
            f.close();
        }
    }
}



/** @brief Destructor de la clase
  *
  */
CFace::~CFace()
{

}


