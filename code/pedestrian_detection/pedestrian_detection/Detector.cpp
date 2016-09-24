#include "Detector.h"

Detector::Detector() {
}

Detector::~Detector(void) {
}

void Detector::TrainDetector(const Mat& trainSample, const Mat& trainLabel, 
	const int* featureSizes, const CvSVMParams& params) {
	this->train(trainSample, trainLabel, Mat(), Mat(), params);
	Mat tem1 = TransSV2Detector();
	Mat tmp2(featureSizes[0], featureSizes[1], CV_32FC(featureSizes[2]), tem1.data);
	tmp2.copyTo(detector);
}

Mat Detector::TransSV2Detector() {
	int numOfFeatures = this->var_all;
	int numSupportVectors = this->get_support_vector_count();
	const CvSVMDecisionFunc *dec = this->decision_func;
	Mat detector(1, numOfFeatures, CV_32FC1, Scalar(0));
	for (int i = 0; i < numSupportVectors; ++i) {
		float alpha = *(dec->alpha+i);
		const float* supportVector = this->get_support_vector(i);
		for(int j = 0; j < numOfFeatures; j++) {
			detector.at<float>(0, j) += alpha * *(supportVector+j);
		}
	}
	return -detector;
}

void Detector::DisplayDetection(Mat im, Mat top) {
	for (int i = 0 ; i < top.size[0] ; ++i) {
		rectangle(im, Point(top.at<float>(i, 1), top.at<float>(i, 0)), 
			Point(top.at<float>(i, 3), top.at<float>(i, 2)), 
			Scalar(0, 0, 255), 2);
	}
	namedWindow("Display detection", WINDOW_AUTOSIZE);
	imshow("Display detection", im); 
			
	waitKey(0);
	destroyWindow("Display detection");
}

void Detector::SetDetector(const Mat det) {
	det.copyTo(this->detector);
}

Mat Detector::GetDetector() {
	return detector;
}

/* Detection is for single scale detection
   Input:
           1. im: A grayscale image in height x width.
           2. ext: The pre-defined HOG extractor.
           3. threshold: The constant threshold to control the prediction.
   Output:
           1. bbox: The predicted bounding boxes in this format (n x 5 matrix):
                                    x11 y11 x12 y12 s1
                                    ... ... ... ... ...
                                    xi1 yi1 xi2 yi2 si
                                    ... ... ... ... ...
                                    xn1 yn1 xn2 yn2 sn
                    where n is the number of bounding boxes. For the ith 
                    bounding box, (xi1,yi1) and (xi2, yi2) correspond to its
                    top-left and bottom-right coordinates, and si is the score
                    of convolution. Please note that these coordinates are
                    in the input image im.
*/
Mat Detector::Detection(const Mat& im, HOGExtractor& ext, const float threshold) {
	
	/* TODO 4: 
       Obtain the HOG descriptor of the input im. And then convolute the linear
	   detector on the HOG descriptor. You may put the score of convolution in 
	   the structure "convolutionScore".
	*/
	//cout<<im.rows<<endl<<im.cols<<endl<<im.channels()<<endl;

	Mat HOGDes = ext.ExtractHOG(im);//256*256*36;detector 15*7*36;
	
	Mat convolutionScore(HOGDes.size[0], HOGDes.size[1], CV_32FC1, Scalar(0));

	// Begin TODO 4
	
	vector<Mat> channelsDes(36);
	vector<Mat> channelsDet(36);
	
	split( HOGDes, channelsDes);
	
	split( detector, channelsDet);
	
	for (int channel=0;channel<36;channel++)
	{
		Mat score(HOGDes.size[0], HOGDes.size[1], CV_32FC1, Scalar(0));
		filter2D(channelsDes[channel], score, CV_32FC1, channelsDet[channel]);
		convolutionScore = convolutionScore + score;
		//cout<<channelsDet[channel].at<float>((channelsDet[channel].size[0]/2,channelsDet[channel].size[1]/2))<<endl;
		//cout<<score.at<float>((HOGDes.size[0]/2,HOGDes.size[1]/2))<<endl;
		
	}
	//cout<<convolutionScore.at<float>(HOGDes.size[0]/2,HOGDes.size[1]/2)<<endl;
	//cout<<"end"<<endl;
	// End TODO 4

	/* TODO 5: 
       Select out those positions where the score is above the threshold. Here,
	   the positions are in ConvolutionScore. To output the coordinates of the
	   bounding boxes, please remember to recover the positions to those in the
	   input image. Please put the predicted bounding boxes and their scores in
	   the below structure "bbox".
	*/
	Mat bbox;
	
	// Begin TODO 5
	
	int xi1, yi1, xi2, yi2;
	float si;
	//cout<<HOGDes.size[0]<<endl<<HOGDes.size[1]<<endl;
	for(int i = 4; i < HOGDes.size[0]-4; i++)
		for(int j = 1; j < HOGDes.size[1]-1; j++) 
		{
			si = convolutionScore.at<float>(i,j);
			if (si > threshold)
			{
				//cout<<si<<">"<<threshold<<endl;
				Mat oneBbox(1, 5, CV_32FC1, Scalar(0));
				int xi1, yi1, xi2, yi2;
				xi1 = 8*(i-7);
				yi1 = 8*(j-3);
				xi2 = xi1 + 16*8;
				yi2 = yi1 + 8*8;
				//cout<<"i="<<i<<", j="<<j<<endl;
				//cout<<xi1<<endl<<yi1<<endl<<xi2<<endl<<yi2<<endl;
				if (xi1<0) xi1 = 0;
				if (yi1<0) yi1 = 0;
				if (xi2>=im.rows) xi2 = im.rows-1;
				if (yi2>=im.cols) yi2 = im.cols-1;
				//cout<<xi1<<endl<<yi1<<endl<<xi2<<endl<<yi2<<endl;
				oneBbox.at<float>(0,0) = (float)xi1;
				oneBbox.at<float>(0,1) = (float)yi1;
				oneBbox.at<float>(0,2) = (float)xi2;
				oneBbox.at<float>(0,3) = (float)yi2;
				oneBbox.at<float>(0,4) = si;
				bbox.push_back(oneBbox);

			}
		}
	
	
	// End TODO 5

	return bbox;
}


/* MultiscaleDetection is for multiscale detection
   Input:
           1. im: A grayscale image in height x width.
           2. ext: The pre-defined HOG extractor.
		   3. scales: The scales for resizeing the image.
		   4. numberOfScale: The number of different scales.
           5. threshold: The constant threshold to control the prediction.
   Output:
           1. bbox: The predicted bounding boxes in this format (n x 5 matrix):
                                    x11 y11 x12 y12 s1
                                    ... ... ... ... ...
                                    xi1 yi1 xi2 yi2 si
                                    ... ... ... ... ...
                                    xn1 yn1 xn2 yn2 sn
                    where n is the number of bounding boxes. For the ith 
                    bounding box, (xi1,yi1) and (xi2, yi2) correspond to its
                    top-left and bottom-right coordinates, and si is the score
                    of convolution. Please note that these coordinates are
                    in the input image im.
*/
Mat Detector::MultiscaleDetection(const Mat& im, HOGExtractor& ext, 
	const float* scales, const int numberOfScale, const float* threshold) {

	/* TODO 6: 
       You should firstly resize the input image by scales 
	   and store them in the structure pyra. 
	*/
	vector<Mat> pyra(numberOfScale);

	// Begin TODO 6
	for (int index=0;index<numberOfScale;index++)
	{
		Mat imgScal;
		int rows = (int)im.rows * scales[index];
		int cols = (int)im.cols * scales[index];
		
		resize(im, imgScal,Size(cols,rows));
		
		pyra[index] = imgScal;
		
	}
	// End TODO 6
	

	/* TODO 7: 
       Perform detection with different scales. Please remember 
	   to transfer the coordinates of bounding box according to 
	   their scales. 
	   You should complete the helper-function  "Detection" and 
	   call it here. All the detected bounding boxes should be 
	   stored in the below structure "bbox".
	*/
	Mat bbox;

	// Begin TODO 7
	for (int index=0;index<numberOfScale;index++)
	{

		Mat bboxScal;
		
		bboxScal = Detection(pyra[index], ext, threshold[index]);
		//change bboxScal coordinates
	    for(int i = 0; i < bboxScal.rows; i++)
		    for(int j = 0; j < 4; j++) 
				bboxScal.at<float>(i,j) /= scales[index];
		bbox.push_back(bboxScal);
	}	
	// End TODO 7

	return bbox;
}

