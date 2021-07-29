/*************************************************************************
	> File Name: convolution_forward.c
	> Author: 
	> Mail: 
	> Created Time: 
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <athread.h>

#include "args.h"
#include "util.h"
#include "normInputs.h"


float *QN , *KN, *VN, *QK;

extern void SLAVE_FUN(funcNorm)(void* inputs);
NormInputs norm_inputs=NULL;

NormInputs create_empty_norm_inputs()
{
	//printf("--------------before Input malloc-----------\n");
    NormInputs norm_inputs = (NormInputs)malloc(sizeof(NormInputsItem));
    //printf("--------------after Input malloc-----------\n");
    norm_inputs->b = 0;
    norm_inputs->n = 0;
    norm_inputs->S = 0;
    norm_inputs->N = 0;
    norm_inputs->QK = NULL;
    return norm_inputs;
}

int multihead_attention(Args_t arg)
{
	const int B = arg->B;
    const int S = arg->S;
    const int D = arg->D;
    const int N = arg->N;
    const float* x = arg->x;
    const float* w = arg->w;
    float* Q = arg->Q;
    float* K = arg->K;
    float* V = arg->V;
    float* QK = arg->QK;
    float* y = arg->y;
	const int PD = D/N;
    memset(Q, 0, sizeof(float)*B*S*D);
    memset(K, 0, sizeof(float)*B*S*D);
    memset(V, 0, sizeof(float)*B*S*D);
    memset(QK, 0, sizeof(float)*B*N*S*S);
    memset(y, 0, sizeof(float)*B*S*D);
	float* QN = (float*)aligned_malloc(sizeof(float)*B*N*S*PD, 128);
	float* KN = (float*)aligned_malloc(sizeof(float)*B*N*S*PD, 128);
	float* VN = (float*)aligned_malloc(sizeof(float)*B*N*S*PD, 128);
    //calculate Q, K, V
    for(int b = 0; b < B; b ++)
    {
        _local_gemm_rcr(x+b*S*D, D, w, D, Q+b*S*D, D, S, D, D);
        _local_gemm_rcr(x+b*S*D, D, w+D*D, D, K+b*S*D, D, S, D, D);
        _local_gemm_rcr(x+b*S*D, D, w+2*D*D, D, V+b*S*D, D, S, D, D);
    }
    _local_trans_head(Q, QN, B, S, D, N);
    _local_trans_head(K, KN, B, S, D, N);
    _local_trans_head(V, VN, B, S, D, N);
#define NI(b,n,s,pd) ((((b)*N+n)*S+s)*PD+pd)
#define QKI(b,n,sh,sl) ((((b)*N+n)*S+sh)*S+sl)
	// QK = Q*KT
	for(int b = 0; b < B; b ++)
		for(int n = 0; n < N; n ++)
			_local_gemm_rcr(QN+NI(b,n,0,0), PD, KN+NI(b,n,0,0), PD, QK+QKI(b,n,0,0), S, S, S, PD);

	double norm = sqrt(PD*1.0);
	for(int i = 0; i < B*N*S*S; i ++)
		QK[i] /= norm;
	for(int b = 0; b < B; b ++)
		for(int n = 0; n < N; n ++){
			/*
			for(int s=0; s < S; s++)
				_local_norm(QK+QKI(b,n,s,0), S);
			*/
			
			if (norm_inputs == NULL) norm_inputs = create_empty_norm_inputs();
		    norm_inputs->S = S;
		    norm_inputs->N = N;
		    norm_inputs->b = b;
		    norm_inputs->n = n;
		    norm_inputs->QK = QK;
		    athread_spawn(funcNorm,norm_inputs);
		    athread_join();  
		    
		}
			
			

	// reuse Q
	memset(QN, 0, sizeof(float)*B*S*D);
	for(int b = 0; b < B; b ++){   
		for(int n = 0; n < N; n ++){
			_local_gemm_rrr(QK+QKI(b,n,0,0), S, VN+NI(b,n,0,0), PD, QN+NI(b,n,0,0), PD, S, PD, S);
        }
    }
    //trans back
	_local_trans_head_back(QN, y, B, S, D, N);
    
	aligned_free(QN);
	aligned_free(KN);
	aligned_free(VN);
    return 0;
}

