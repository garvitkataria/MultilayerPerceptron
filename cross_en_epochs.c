#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
typedef struct node
{
	int n;
	double bias, wts[20];

}node;

int trainData[2000][17] = {0};
double biasInput, biasHidden;
node in[16], hidden[7], out[10];


double sigmoid_func(double x)
{
	return (double)((1.0)/(1+exp(-1*x)));
}

double deriv_function(double x)
{
	return (double)(sigmoid_func(x)*(1-sigmoid_func(x)));
}

double randomize_weights()
{
	return ((((double)(rand()) / (double)(RAND_MAX))*6)-3)/100.0;
}

void load_train_data_csv()
{
	FILE *train_csv = fopen("train.csv", "r");
	for (int i = 0; i < 2000; ++i)
	{
		fscanf(train_csv, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", &trainData[i][0], &trainData[i][1], &trainData[i][2], &trainData[i][3], &trainData[i][4], &trainData[i][5], &trainData[i][6], &trainData[i][7], &trainData[i][8], &trainData[i][9], &trainData[i][10], &trainData[i][11], &trainData[i][12], &trainData[i][13], &trainData[i][14], &trainData[i][15], &trainData[i][16]);
	}
	fclose(train_csv);
}

double MatrixProduct(node ne, double *x)
{	
	double ans=0;
	for (int i = 0; i < ne.n; ++i)
	{
		ans += ne.wts[i]*x[i];
	}
	return ans+ne.bias;
}

void init(){
	for (int i = 0; i < 16; ++i)
	{
		in[i].n = 1;
		for (int j = 0; j < in[i].n; ++j)
		{
			in[i].wts[j] = 1;		
		}
		in[i].bias = 1;		
	}

	for (int i = 0; i < 7; ++i)
	{
		hidden[i].n = 16;
		for (int j = 0; j < hidden[i].n; ++j)
		{
			hidden[i].wts[j] = randomize_weights();		
		}
		hidden[i].bias = 1;		
	}

	for (int i = 0; i < 10; ++i)
	{
		out[i].n = 7;
		for (int j = 0; j < out[i].n; ++j)
		{
			out[i].wts[j] = randomize_weights();		
		}
		out[i].bias = 1;		
	}
	biasInput = randomize_weights();
	biasHidden = randomize_weights();
}



int main()
{
	srand((unsigned int)time(NULL));
	load_train_data_csv();
	init();
	double fderv,sum,delta[20],eta = 0.01;
	int tk=0;
	double netk[20], netj[20], dif;
	double result_in[2000][16]={0}, result_hid[2000][6]={0}, result_out[2000][10]={0};
	int epoch = 1;
	while(epoch<1000)
	{
		for (int i = 0; i < 2200; ++i)
		{
			double in_out[16]={0};

			for (int j = 0; j < 16; ++j)
			{
				in_out[j] = (trainData[i][j+1])*1.0;					
			}

			double hidden_out[7] = {0};
			for (int j = 0; j < 7; ++j)
			{
				netj[j] = MatrixProduct(hidden[j], in_out)+biasInput;
				hidden_out[j] = sigmoid_func(netj[j]);
			}

			double out_out[10] = {0};
			for (int j = 0; j < 10; ++j)
			{
				netk[j] = MatrixProduct(out[j], hidden_out)+biasHidden;
				out_out[j] = sigmoid_func(netk[j]); 
			}
			for(int k = 0; k < 10; ++k)	
			{
				tk = 0;
				if((k+1) == trainData[i][0])
				{
					tk = 1;
				}
				delta[k] = ((double)tk-out_out[k]);
				for(int j=0; j<7; j++)
				{
					dif = delta[k]*hidden_out[j]*eta;
					out[k].wts[j] += dif;
				}
				dif = delta[k]*biasHidden*eta;
				biasHidden += dif;
			}

			for(int j=0;j<7;j++)
			{
				fderv = deriv_function(netj[j]);
				for (int _i = 0; _i < 16; ++_i)
				{
					sum = 0;
					for(int r=0; r<10; r++)
					{
						sum += delta[r] * out[r].wts[j];
					}
					dif = sum * eta * fderv * in_out[_i];
					hidden[j].wts[_i] += dif;
				}
				dif = sum * eta * fderv * biasInput;
				biasInput+=dif;
			}
		}
		epoch++;
	}


	FILE *train_csv = fopen("test.csv", "r");
	int count =0,p;
	double max;
	double in_out[16]={0}; 
	for (int i = 0; i < 100; ++i)
	{
		fscanf(train_csv, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", &trainData[i][0], &trainData[i][1], &trainData[i][2], &trainData[i][3], &trainData[i][4], &trainData[i][5], &trainData[i][6], &trainData[i][7], &trainData[i][8], &trainData[i][9], &trainData[i][10], &trainData[i][11], &trainData[i][12], &trainData[i][13], &trainData[i][14], &trainData[i][15], &trainData[i][16]);
			for (int j = 0; j < 16; ++j)
			{
				in_out[j] = (in[j].wts[0]*trainData[i][j+1]);			
			}

			double hidden_out[7] = {0};
			for (int j = 0; j < 7; ++j)
			{
				netj[j] = MatrixProduct(hidden[j], in_out)+biasInput;
				hidden_out[j] = sigmoid_func(netj[j]);
			}

			double out_out[10] = {0};
			max = out_out[0];
			p=0;
			for (int j = 0; j < 10; ++j)
			{
				netk[j] = MatrixProduct(out[j], hidden_out)+biasHidden;
				out_out[j] = sigmoid_func(netk[j]); 
				if(out_out[j] > max){
					max = out_out[j];
					p=j;
				}
			}
			if((p+1) == (int)trainData[i][0]){
				count++;
			}
	}
	fclose(train_csv);
	printf("Accuracy = %f\n", count/100.0*100.0);
	return 0;
}

