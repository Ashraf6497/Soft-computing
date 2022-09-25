#include<stdio.h>
#include<conio.h>
#include<math.h>
#include<stdlib.h>
int main()
{
	int i,j,k,p,L,N,P,T,TIO,M,count=0; /*TIO = total num of input and output neurons; T = test patterns;M = num of hidden neurons*/
	double a[100][100],V[100][100],W[100][100],E[100][100],dw[100][100],dv[100][100],eta,max,min,mse; /*E = Target - output*/
	double IH[100][100],OH[100][100],IO[100][100],OO[100][100],I[100][100],Ta[100][100],alpha; /* Ta = target matrix*/
	double Tmax[100],Tmin[100],AO[100][100];
	mse = 1;
	eta = 0.5;
	alpha = 0.7;
	M=31;
	FILE *ip,*op1,*op2;
	/*reading data from input*/
	ip=fopen("input_1.txt","r");
	op1=fopen("output_1t.txt","w");
	op2=fopen("output_2t.txt","w");
	fscanf(ip,"%d",&L);
	fscanf(ip,"%d",&N);
	fscanf(ip,"%d",&P);
	fscanf(ip,"%d",&T);
	fprintf(op1,"count				MSE\n");
	TIO = L+N;
	for(i=1;i<=P+T;i++)
	{
		for(j=1;j<=TIO;j++)
		{
			fscanf(ip,"%lf",&a[i][j]);
		}
	}
	/*target matrix*/
	for(k=L+1;k<=L+N;k++)
	{
		for(p=1;p<=P+T;p++)
		{
			Ta[k-L][p] = a[p][k];
		}
	}
	fprintf(op2,"Target matrix : \n");
	for(p=P+1;p<=P+T;p++)
	{
		for(k=L+1;k<=L+N;k++)
		{
			fprintf(op2,"%lf ",Ta[k-L][p]);
		}
		fprintf(op2,"\n");
	}
	/*obtaining maximum and minimum matrices for targets*/
	for(k=L+1;k<=L+N;k++)
	{
		Tmax[k-L] = Ta[k-L][1];
		for(p=1;p<=P;p++)
		{
			if(Ta[k-L][p]>Tmax[k-L])
			{
				Tmax[k-L] = Ta[k-L][p];
			}
		}
	}
	for(k=L+1;k<=L+N;k++)
	{
		Tmin[k-L] = Ta[k-L][1];
		for(p=1;p<=P;p++)
		{
			if(Ta[k-L][p]<Tmin[k-L])
			{
				Tmin[k-L] = Ta[k-L][p];
			}
		}
	}
	/*obtaining max and min from columns in datasets for normalization*/
	for(j=1;j<=TIO;j++)
	{
		max = a[1][j];
		min = a[1][j];
		for(i=1;i<=P+T;i++)
		{
			if(a[i][j]>max)
			{
				max = a[i][j];
			}
			if(a[i][j]<min)
			{
				min = a[i][j];
			}
		}
		/*normalising of datasets for log sigmoid function*/
		for(i=1;i<=P+T;i++)
		{
			a[i][j] = 0.1 + (0.8 * ((a[i][j]-min)/(max-min)));
		}
	}
	/*Input matrix*/
	for(i=1;i<=L;i++)
	{
		for(p=1;p<=P;p++)
		{
			I[i][p]=a[p][i];
		}
	}
	/*target matrix*/
	for(k=L+1;k<=L+N;k++)
	{
		for(p=1;p<=P+T;p++)
		{
			Ta[k-L][p] = a[p][k];
		}
	}
	/*random initialization of connecting weights*/
	/*for V connecting weight b/w input and hidden*/
	for(i=0;i<=L;i++)
	{
		for(j=1;j<=M;j++)
		{
			V[i][j] = sin(rand());
		}
	}
	/*for W connecting weight b/w hidden and output*/
	for(j=0;j<=M;j++)
	{
		for(k=1;k<=N;k++)
		{
			W[j][k]= sin(rand());
		}
	}
	while(mse > 0.001 && count<50000)
	{
		mse = 0;
		/*hidden input matrix*/
		for(p=1;p<P+1;p++)
		{
			for(j=1;j<=M;j++)
			{
				IH[j][p] = 0;
				for(i=1;i<=L;i++)
				{
					IH[j][p]=IH[j][p] + (I[i][p] * V[i][j]);
				}
				IH[j][p] = IH[j][p] + (1*V[0][j]);
			}
		}
		/*hidden Transform matrix*/
		for(p=1;p<P+1;p++)
		{
			for(j=1;j<=M;j++)
			{
				OH[j][p] = 1/(1+exp(-IH[j][p]));
			}
		}
		/*o/p layer input matrix*/
		for(p=1;p<P+1;p++)
		{
			for(k=1;k<=N;k++)
			{
				IO[k][p] = 0;
				for(j=1;j<=M;j++)
				{
					IO[k][p]=IO[k][p] + (OH[j][p] * W[j][k]);
				}
				IO[k][p] = IO[k][p] + (1*W[0][k]);
			}
		}
		/*output transformation matrix*/
		for(p=1;p<P+1;p++)
		{
			for(k=1;k<=N;k++)
			{
				OO[k][p] = 1/(1+exp(-IO[k][p]));
			}
		}
		/*Calculation of error*/
		for(k=1;k<=N;k++)
		{
			for(p=1;p<=P;p++)
			{
				E[k][p]=Ta[k][p]-OO[k][p];
				mse = mse + (0.5*E[k][p]*E[k][p]);
			}
		}
		mse = mse/P;
		/*updating of W*/
		for(j=0;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				dw[j][k] = 0;
				for(p=1;p<=P;p++)
				{
					dw[j][k] = dw[j][k] + ((E[k][p])*(OO[k][p])*(1-OO[k][p])*(OH[j][p]));
				}
			}
		}
		for(j=0;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				dw[j][k] = (eta/P)*dw[j][k];
			}
		}
		/*updating of V*/
		for(i=0;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				dv[i][j] = 0;
				for(p=1;p<=P;p++)
				{
					for(k=1;k<=N;k++)
					{
						dv[i][j] = dv[i][j] + ((E[k][p])*(OO[k][p])*(1-(OO[k][p]))*(W[j][k])*(OH[j][p])*(1-OH[j][p])*(I[i][p]));
					}
				}
			}
		}
		for(i=0;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				dv[i][j] = (eta/(P*N))*dv[i][j];
			}
		}
		/*Wnew*/;
		for(j=0;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				W[j][k] = W[j][k] + (alpha * dw[j][k]);
			}
		}
		/*Vnew*/
		for(i=0;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				V[i][j] = V[i][j] + (alpha * dv[i][j]);
			}
		}	
		count++;
		printf(" count = %d and MSE = %lf \n\n",count,mse);
		fprintf(op1,"%d				%lf\n",count,mse);
	}
	printf("TEST CASE: \n");
	/*printing of Vfinal*/
	printf("V matrix is : \n");
	for(i=0;i<=L;i++)
	{
		for(j=1;j<=M;j++)
		{
			printf("%lf  ",V[i][j]);
		}
		printf("\n");
	}
	/*printing of Wfinal*/
	printf("W matrix is : \n");	
	for(j=0;j<=M;j++)
	{
		for(k=1;k<=N;k++)
		{
			printf("%lf  ",W[j][k]);
		}
		printf("\n");
	}
	/*test case inputs*/
	for(i=1;i<=L;i++)
	{
		for(p=P+1;p<=P+T;p++)
		{
			I[i][p]=a[p][i];
		}
	}
	/*hidden input matrix*/
		for(p=P+1;p<=P+T;p++)
		{
			for(j=1;j<=M;j++)
			{
				IH[j][p] = 0;
				for(i=1;i<=L;i++)
				{
					IH[j][p]=IH[j][p] + (I[i][p] * V[i][j]);
				}
				IH[j][p] = IH[j][p] + (1*V[0][j]);
			}
		}
		/*hidden Transform matrix*/
		for(p=P+1;p<=P+T;p++)
		{
			for(j=1;j<=M;j++)
			{
				OH[j][p] = 1/(1+exp(-IH[j][p]));
			}
		}
		/*o/p layer input matrix*/
		for(p=P+1;p<=P+T;p++)
		{
			for(k=1;k<=N;k++)
			{
				IO[k][p] = 0;
				for(j=1;j<=M;j++)
				{
					IO[k][p]=IO[k][p] + (OH[j][p] * W[j][k]);
				}
				IO[k][p] = IO[k][p] + (1*W[0][k]);
			}
		}
		/*output transformation matrix*/
		for(p=P+1;p<=P+T;p++)
		{
			for(k=1;k<=N;k++)
			{
				OO[k][p] = 1/(1+exp(-IO[k][p]));
			}
		}
		/*Actual output*/
		for(k=L+1;k<=L+N;k++)
		{
			for(p=P+1;p<=P+T;p++)
			{
				AO[k][p]= Tmin[k-L]+ ((Tmax[k-L]-Tmin[k-L])/0.8)*(OO[k-L][p]-0.1);
			}
		}
		printf("Actual output is: \n");
		for(p=P+1;p<=P+T;p++)
		{
			for(k=L+1;k<=L+N;k++)
			{
				printf("%lf ",AO[k][p]);
			}
			printf("\n");
		}
		fprintf(op2,"Actual output is: \n");
		for(p=P+1;p<=P+T;p++)
		{
			for(k=L+1;k<=L+N;k++)
			{
				fprintf(op2,"%lf ",AO[k][p]);
			}
			fprintf(op2,"\n");
		}
		fprintf(op2,"\n");
		/*Calculation of error*/
		for(k=1;k<=N;k++)
		{
			for(p=P+1;p<=P+T;p++)
			{
				E[k][p]=Ta[k][p]-OO[k][p];
				mse = mse + (0.5*E[k][p]*E[k][p]);
			}
		}
		mse = mse/T;
		printf("mse = %lf \n",mse);
		fprintf(op2,"mse = %lf \n",mse);
		printf("error matrix: \n");
		for(k=1;k<=N;k++)
		{
			for(p=P+1;p<=P+T;p++)
			{
				printf("%lf  ",E[k][p]);
			}
			printf("\n");
		}
	return 0;
}
