#include "HOGExtractor.h"


HOGExtractor::HOGExtractor(const int bins, const int cells, const int blocks) {
	this->bins = bins;
	this->cells = cells;
	this->blocks = blocks;
}


HOGExtractor::~HOGExtractor(void) {
}

int HOGExtractor::GetBins() {
	return this->bins;
}
int HOGExtractor::GetCells() {
	return this->cells;
}
int HOGExtractor::GetBlocks() {
	return this->blocks;
}


/* ExtractHOG is for extracting HOG descriptor of an image
   Input:
           im: A grayscale image in height x width.
   Output:
           HOGBlock: The HOG descriptor of the input image.
*/
Mat HOGExtractor::ExtractHOG(const Mat& im) {
	// Pad the im in order to make the height and width the multiplication of
    // the size of cells.

	int height = im.rows;
	int width = im.cols;
	int padHeight = height % cells == 0 ? 0 : (cells - height % cells);
    int padWidth = width % cells == 0 ? 0 : (cells - width % cells);
	Mat paddedIm(height+padHeight, width+padWidth, CV_32FC1, Scalar(0));
	Range imRanges[2];
	imRanges[0] = Range(0, height);
	imRanges[1] = Range(0, width);
	im.copyTo(paddedIm(imRanges));
	height = paddedIm.rows;
	width = paddedIm.cols;

	/* TODO 1: 
       Compute the horizontal and vertical gradients for each pixel. Put them 
	   in gradX and gradY respectively. In addition, compute the angles (using
	   atan2) and magnitudes by gradX and gradY, and put them in angle and 
	   magnitude. 
	*/

	Mat hx(1, 3, CV_32FC1, Scalar(0));
	hx.at<float>(0, 0) = -1;
	hx.at<float>(0, 1) = 0;
	hx.at<float>(0, 2) = 1;
	Mat hy = -hx.t();

	Mat gradX(height, width, CV_32FC1, Scalar(0));
	Mat gradY(height, width, CV_32FC1, Scalar(0));
	Mat angle(height, width, CV_32FC1, Scalar(0));
	Mat magnitude(height, width, CV_32FC1, Scalar(0));
	float pi = 3.1416;
	
	// Begin TODO 1
	
	//cout<<im.at<uchar>(4,6)<<endl<<im.at<uchar>(4,8)<<endl;
	filter2D(paddedIm, gradX, CV_32F, hx);
	filter2D(paddedIm, gradY, CV_32F, hy);
	
	//cout<<gradX.at<float>(4,7)<<endl<<gradY.at<float>(4,7)<<endl<<endl<<endl;
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			
			float x = gradX.at<float>(i,j);
			float y = gradY.at<float>(i,j);
			float tempAng = fastAtan2(y, x)*pi/180;
			if (tempAng > pi)
				tempAng = tempAng - 2*pi;
			angle.at<float>(i,j) = tempAng;
			//cout<<angle.at<float>(i,j)<<'\t';
			magnitude.at<float>(i,j) = sqrt(x*x + y*y);
			//cout<<i<<j<<endl;
		}
		
	}

	

    // End TODO 1
	

	/* TODO 2:
	   Construct HOG for each cells, and put them in HOGCell. numberOfVerticalCell
	   and numberOfHorizontalCell are the numbers of cells in vertical and 
	   horizontal directions.
	   You should construct the histogram according to the bins. The bins range
	   from -pi to pi in this project, and the interval is given by
	   (2*pi)/bins.
	*/
		
	int numberOfVerticalCell = height / cells;
	int numberOfHorizontalCell = width / cells;
	Mat HOGCell(numberOfVerticalCell, numberOfHorizontalCell, 
		CV_32FC(bins), Scalar(0));
	float piInterval = 2 * pi / bins;
	
	// Begin TODO 2
	
	for(int i = 0; i < numberOfVerticalCell; i++)
		for(int j = 0; j < numberOfHorizontalCell; j++) 
		{
			Mat imCell = paddedIm(Range(8*i,8*i+7),Range(8*j,8*j+7));
			for(int m = 0; m < 8; m++)
				for(int n = 0; n < 8; n++)
				{
					float tempAng = angle.at<float>(i*8+m,j*8+n);
					int tempBin = ceil(tempAng/piInterval + 3.5);				
					//cout<<tempBin<<" ";
					HOGCell.at<Vec<float, 9>>(i,j)[tempBin] += magnitude.at<float>(i*8+m,j*8+n);
				}
			//cout<<endl;
		}
	


	// End TODO 2


	/* TODO 3:
	   Concatenate HOGs of the cells within each blocks and normalize them.
	   The result should be stored in HOGBlock, where numberOfVerticalBlock and
	   numberOfHorizontalBlock are the number of blocks in vertical and
	   horizontal directions.
	*/
	int numberOfVerticalBlock = numberOfVerticalCell - 1;
	int numberOfHorizontalBlock = numberOfHorizontalCell - 1;
	Mat HOGBlock(numberOfVerticalBlock, numberOfHorizontalBlock, 
		CV_32FC(blocks*blocks*bins), Scalar(0));
	
	// Begin TODO 3
	Vec<float, 36> vBlock;
	for(int i = 0; i < numberOfVerticalBlock; i++)
		for(int j = 0; j < numberOfHorizontalBlock; j++) 
		{	
			
			for(int k=0; k<9; k++)
			{
				vBlock[k] = HOGCell.at<Vec<float, 9>>(i,j)[k];

				vBlock[k+bins] = HOGCell.at<Vec<float, 9>>(i,j+1)[k];

				vBlock[k+bins*2] = HOGCell.at<Vec<float, 9>>(i+1,j)[k];

				vBlock[k+bins*3] = HOGCell.at<Vec<float, 9>>(i+1,j+1)[k];
			}



			//Normalization L2_norm
			//float normBlock = norm(vBlock);
			//normBlock = sqrt(normBlock*normBlock+FLT_EPSILON*FLT_EPSILON);
			//for(int k=0; k<36; k++)
			//	vBlock[k] /= normBlock;


			//Normalization L1_norm
			//float normBlock = norm(vBlock, NORM_L1);
			//normBlock = normBlock + FLT_EPSILON;
			//for(int k=0; k<36; k++)
			//	vBlock[k] /= normBlock;
			//cout<<norm(vBlock)<<" "<<norm(vBlock, NORM_L1)<<endl;

			//Normalization L1_Sqrt
			float normBlock = norm(vBlock, NORM_L1);
			normBlock = normBlock + FLT_EPSILON;
			for(int k=0; k<36; k++)
				vBlock[k] = sqrt(vBlock[k]/normBlock);
			//cout<<norm(vBlock)<<endl;

			////Normalization L2_Hys
			//float normBlock = norm(vBlock);
			//normBlock = sqrt(normBlock*normBlock+FLT_EPSILON*FLT_EPSILON);
			//for(int k=0; k<36; k++)
			//{
			//	vBlock[k] /= normBlock;
			//	if (vBlock[k] > 0.2)
			//		vBlock[k] = 0.2;
			//}
			//cout<<norm(vBlock)<<endl;

			HOGBlock.at<Vec<float, 36>>(i,j) = vBlock;
		}

	// End TODO 3

			

	return HOGBlock;
}