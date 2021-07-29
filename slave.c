#include <slave.h>
#include <math.h>
#include <simd.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

#include "args.h"
#include "inputs.h"
#include "normInputs.h"

extern float *QN, *KN, *VN, *QK;

void wait_reply(volatile unsigned int *reply, int n) {
    while (*reply != n) {};
}

void funcNorm(NormInputs inputs) {
	//printf("---------\n");
	volatile unsigned int get_reply=0;
	volatile unsigned int put_reply=0;
	const int id = athread_get_id(-1);
	int b = inputs->b;
	int n = inputs->n;
	int S = inputs->S;
	int N = inputs->N;
	float *QK = inputs->QK;
	float QKL[S];
	int Sp, s_start;
#define QKI(b,n,sh,sl) ((((b)*N+n)*S+sh)*S+sl)

	int Sp1=floor(S/64);
	int rem = S%64;
	if (id < rem) {
		Sp = Sp1+1;
		s_start = id*Sp;
	}
	else {
		Sp = Sp1;
		s_start = Sp*id+rem;
	}

	if (Sp!=0){
		
		for(int s = s_start; s < s_start+Sp; s ++){
			//_local_norm(QK+QKI(b,n,s,0), S);
			//printf("=======QK[QKI]=%f====\n",QK[QKI(b,n,s,0)]);
			athread_get(PE_MODE,&QK[QKI(b,n,s,0)],&QKL[0],sizeof(float)*S,&get_reply,0,0,0);
			wait_reply(&get_reply, 1);
			get_reply=0;
			//printf("----QKL[0]=%f,QKL[S-1]=%f---\n",QKL[0],QKL[S-1]);
			double sum = 0.0;
			for(int i = 0;i < S; i ++)
				sum += QKL[i];
			for(int i = 0;i < S;i ++)
				QKL[i] /= sum;
			athread_put(PE_MODE,&QKL[0],&QK[QKI(b,n,s,0)],sizeof(float)*S,&put_reply,0,0);
			wait_reply(&put_reply, 1);
			put_reply=0;
			//printf("---------s=%d--------\n", s);
		}
	}
	
}

void funcRCR(Inputs inputs)
{
	volatile unsigned int get_reply=0;
	volatile unsigned int put_reply=0;
	const int id = athread_get_id(-1);
	float* A = inputs->A;
	float* B = inputs->B;
	float* C = inputs->C;
	int LDA = inputs->LDA;
	int LDB = inputs->LDB;
	int LDC = inputs->LDC;
	int M = inputs->M;
	int N = inputs->N;
	int K = inputs->K;
	int Mp, m_start;
	int Np, n_start;
	int Kp, k_start;

	int Mp1=floor(M/64);
	int rem = M%64;
	if (id < rem) {
		Mp = Mp1+1;
		m_start = id*Mp;
	}
	else {
		Mp = Mp1;
		m_start = Mp*id+rem;
	}

	if (Mp!=0){

		//float A_l[(Mp-1)*LDA+K];
		float A_l[K];
		float C_l[N];


		float B_l[K];

		for(int i = 0;i < Mp; i ++)
		{
			athread_get(PE_MODE,&C[(m_start+i)*LDC],&C_l[0],sizeof(float)*N,&get_reply,0,0,0);
			while (get_reply!=1);
	    	wait_reply(&get_reply, 1);
			get_reply=0;
			memset(C_l, 0, N*sizeof(float));

			athread_get(PE_MODE,&A[(m_start+i)*LDA],&A_l[0],sizeof(float)*K,&get_reply,0,0,0);
			//while (get_reply!=1);
	    	wait_reply(&get_reply, 1);
			get_reply=0;
		
        	for(int j = 0; j < N; j ++)
        	{
            	athread_get(PE_MODE,&B[j*LDB],&B_l[0],sizeof(float)*K,&get_reply,0,0,0);
    			wait_reply(&get_reply, 1);
				get_reply=0;
            	//printf("------------in j=%d loop---------\n",j_start+j);
            	for(int k = 0; k < K; k +=4){
            		
            		floatv4 va;
            		floatv4 vb;

            		simd_load(va, &A_l[k]);
            		simd_load(vb, &B_l[k]);

            		floatv4 vab = va*vb;
            		float ab[4];
            		simd_store(vab,ab);

                	C_l[j] += (ab[0]+ab[1]+ab[2]+ab[3]);
                	
                	
                	//C_l[j]+=A_l[k]*B_l[k];
                	
            	}
            	/*
            	buff[j%32]=C_l[j];
            	if (j%32==31) {
            		//printf("-----full put,j=%d---\n",j);
            		athread_put(PE_MODE,&buff[0],&C[(m_start+i)*LDC+j-31],sizeof(float)*32,&put_reply,0,0);
        			//while (put_reply!=1);
            		wait_reply(&put_reply, 1);
					put_reply=0;
					//printf("====full put===\n");
            	}
            	else if (j==N) {
            		//printf("-----last put,j=%d---\n",j);
            		athread_put(PE_MODE,&buff[0],&C[(m_start+i)*LDC+j-(j%32)],sizeof(float)*((j%32)+1),&put_reply,0,0);
        			//while (put_reply!=1);
            		wait_reply(&put_reply, 1);
					put_reply=0;
					//printf("=====last put====\n");
            	}
            	*/
				
        	}

			athread_put(PE_MODE,&C_l[0],&C[(m_start+i)*LDC],sizeof(float)*N,&put_reply,0,0);
        	//while (put_reply!=1);
            wait_reply(&put_reply, 1);
			put_reply=0;
        	
	        
		}
	}
}


void funcRRR(Inputs inputs)
{
	volatile unsigned int get_reply=0;
	volatile unsigned int put_reply=0;
	const int id = athread_get_id(-1);
	float* A = inputs->A;
	float* B = inputs->B;
	float* C = inputs->C;
	int LDA = inputs->LDA;
	int LDB = inputs->LDB;
	int LDC = inputs->LDC;
	int M = inputs->M;
	int N = inputs->N;
	int K = inputs->K;
	int Mp, m_start;
	int Np, n_start;
	int Kp, k_start;

	int Mp1=floor(M/64);
	int rem = M%64;
	if (id < rem) {
		Mp = Mp1+1;
		m_start = id*Mp;
	}
	else {
		Mp = Mp1;
		m_start = Mp*id+rem;
	}

	if (Mp!=0){

		//float A_l[(Mp-1)*LDA+K];
		float A_l[K];
		float C_l[N];


		float B_l[4]; //K*LDB+N
		float buff[32];

		for(int i = 0;i < Mp; i ++)
		{
			//athread_get(PE_MODE,&C[(m_start+i)*LDC],&C_l[0],sizeof(float)*N,&get_reply,0,0,0);
			//while (get_reply!=1);
	    	//wait_reply(&get_reply, 1);
			//get_reply=0;
			memset(C_l, 0, sizeof(C_l));

			athread_get(PE_MODE,&A[(m_start+i)*LDA],&A_l[0],sizeof(float)*K,&get_reply,0,0,0);
			//while (get_reply!=1);
	    	wait_reply(&get_reply, 1);
			get_reply=0;
		
        	for(int j = 0; j < N; j ++)
        	{
            	//printf("------------in j=%d loop---------\n",j_start+j);
            	if (K>4){
	            	for(int k = 0; k < K; k +=4){
	            		athread_get(PE_MODE,&B[k*LDB+j],&B_l[0],sizeof(float)*4,&get_reply,0,sizeof(float)*LDB-sizeof(float),sizeof(float));
	    				wait_reply(&get_reply, 1);
						get_reply=0;
						/*
						athread_get(PE_MODE,&B[k*LDB+j],&B_l[0],sizeof(float),&get_reply,0,0,0);
	    				wait_reply(&get_reply, 1);
						get_reply=0;
						athread_get(PE_MODE,&B[(k+1)*LDB+j],&B_l[1],sizeof(float),&get_reply,0,0,0);
	    				wait_reply(&get_reply, 1);
						get_reply=0;
						athread_get(PE_MODE,&B[(k+2)*LDB+j],&B_l[2],sizeof(float),&get_reply,0,0,0);
	    				wait_reply(&get_reply, 1);
						get_reply=0;
						athread_get(PE_MODE,&B[(k+3)*LDB+j],&B_l[3],sizeof(float),&get_reply,0,0,0);
	    				wait_reply(&get_reply, 1);
						get_reply=0;
						*/
						floatv4 va, vb;
						vb=simd_load(vb,&B_l[0]);
						va=simd_load(va,&A_l[k]);
						floatv4 vab = va*vb;
						float ab[4];
						simd_store(vab, ab);

						C_l[j] += (ab[0] + ab[1] + ab[2] + ab[3]);

	            	}
	            }
	            else{
	            	for(int k = 0; k < K; k ++){
	            		athread_get(PE_MODE,&B[k*LDB+j],&B_l[0],sizeof(float),&get_reply,0,0,0);
	    				wait_reply(&get_reply, 1);
						get_reply=0;
						/*
						floatv4 vb;
						simd_load(vb, &B_l[0]);
						floatv4 va=(A_l[k],A_l[k+1],A_l[k+2],A_l[k+3]);
						floatv4 vab = va*vb;
						float ab[4];
						simd_store(vab, ab);

						C_l[j] += (ab[0] + ab[1] + ab[2] + ab[3]);
						*/
	                	C_l[j] += A_l[k]*B_l[0];
	                }
	            }
            	/*
            	buff[j%32]=C_l[j];
            	if (j%32==31) {
            		//printf("-----full put,j=%d---\n",j);
            		athread_put(PE_MODE,&buff[0],&C[(m_start+i)*LDC+j-31],sizeof(float)*32,&put_reply,0,0);
        			//while (put_reply!=1);
            		wait_reply(&put_reply, 1);
					put_reply=0;
					//printf("====full put===\n");
            	}
            	else if (j==N) {
            		//printf("-----last put,j=%d---\n",j);
            		athread_put(PE_MODE,&buff[0],&C[(m_start+i)*LDC+j-(j%32)],sizeof(float)*((j%32)+1),&put_reply,0,0);
        			//while (put_reply!=1);
            		wait_reply(&put_reply, 1);
					put_reply=0;
					//printf("=====last put====\n");
            	}
            	*/
				
        	}
        	athread_put(PE_MODE,&C_l[0],&C[(m_start+i)*LDC],sizeof(float)*N,&put_reply,0,0);
        	//while (put_reply!=1);
            wait_reply(&put_reply, 1);
			put_reply=0;
        	
	        
		}
	}
}
