/*************************************************************************
	> File Name: convolution_forward.c
	> Author: 
	> Mail: 
	> Created Time: 
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <athread.h>
#include "function.h"
#include "args.h"
#include "util.h"
#include <sys/types.h>//所有这些类型在 ILP32 编译环境中保持为 32 位值，并会在 LP64 编译环境中增长为 64 位值。用到的头文件。 
#include <sys/stat.h>//使用stat函数最多的可能是ls-l命令，用其可以获得有关一个文件的所有信息。
#include <fcntl.h>//用来操作文件描述符的一些特性



extern void SLAVE_FUN(par_multihead_attn)(); //declare slave parallel method
extern void SLAVE_FUN(gemm_rcr1)();
extern void SLAVE_FUN(gemm_rcr2)();
extern void SLAVE_FUN(gemm_rrr)();
extern void SLAVE_FUN(gemm_rrr2)();
extern void SLAVE_FUN(sa_norm1)();
extern void SLAVE_FUN(sa_norm2)();

extern int multihead_attention(Args_t arg);
extern int multihead_attention(Args_t arg);
extern int multihead_attention(Args_t arg);




const float* x;
const float* w;
float* Q;
float* K;
float* V;
float* QK;
float* QN;
float* KN;
float* VN;
float* VNT;

typedef struct Dt
{
    int b; // batch
    int S; // sequence length
    int D; // vector size
	int n; //head number
	int N;//heads
	int B;
}Dts, *Dt_a;

//原题函数
static void _local_gemm_rcr(const float* A, const int LDA, const float* B, const int LDB, float* C, const int LDC, int M, int N, int K)
{
    for(int i = 0;i < M; i ++)
        for(int j = 0; j < N; j ++)
            for(int k = 0; k < K; k ++)
                C[i*LDC+j] += A[i*LDA+k]*B[k+j*LDB];
}

static void _local_gemm_rrr(const float* A, const int LDA, const float* B, const int LDB, float* C, const int LDC, int M, int N, int K)
{
    for(int i = 0;i < M; i ++)
        for(int j = 0; j < N; j ++)
            for(int k = 0; k < K; k ++)
                C[i*LDC+j] += A[i*LDA+k]*B[k*LDB+j];
}



static void _local_trans_head(float* src, float* dst, int B, int S, int D, int N)
{
    int pD = D/N;
#define SRC(b, s, d) src[b*S*D+s*D+d]
#define DST(b, n, s, pd) dst[b*N*S*pD + n*S*pD + s*pD + pd]
    for(int b = 0; b < B; b ++)
        for(int n = 0; n < N; n ++)
            for(int s = 0; s < S; s ++)
                for(int pd = 0; pd < pD; pd ++)
                    DST(b,n,s,pd) = SRC(b,s,n*pD+pd);
}

static void _local_trans_head_back(float* src, float* dst, int B, int S, int D, int N)
{
    int pD = D/N;
#define D3(b, s, d) dst[b*S*D+s*D+d]
#define D4(b, n, s, pd) src[b*N*S*pD + n*S*pD + s*pD + pd]
    for(int b = 0; b < B; b ++)
        for(int n = 0; n < N; n ++)
            for(int s = 0; s < S; s ++)
                for(int pd = 0; pd < pD; pd ++)
					D3(b,s,n*pD+pd) = D4(b,n,s,pd);
}


static void _local_norm(float* buf, int len)
{
	double sum = 0.0f;
	for(int i = 0;i < len; i ++)
		sum += buf[i];
	for(int i = 0;i < len;i ++)
		buf[i] /= sum;
}

static void _print_buf(float* buf, int len, const char* name)
{
	printf("====%s\n", name);
	for(int i = 0; i < 10 && i < len; i ++)
		printf("%f ", buf[i]);
	printf("\n");
}


int multihead_attention(Args_t arg)
{
	//printf("mult is begining!,%d\n",sizeof(float));
	//初始化
	Dt_a dt =(Dt_a)malloc(sizeof(Dts));
	int temp;
	const int B = arg->B;
    const int S = arg->S;
    const int D = arg->D;
    const int N = arg->N;
	dt->D = D;
	dt->S = S;
	dt->N = N;
	dt->B = B;
    x = arg->x;
    w = arg->w;
    Q = arg->Q;
    K = arg->K;
    V = arg->V;
    QK = arg->QK;
    float* y = arg->y;
	const int PD = D/N;
    memset(Q, 0, sizeof(float)*B*S*D);
    memset(K, 0, sizeof(float)*B*S*D);
    memset(V, 0, sizeof(float)*B*S*D);
    memset(QK, 0, sizeof(float)*B*N*S*S);
    memset(y, 0, sizeof(float)*B*S*D);
	QN = (float*)aligned_malloc(sizeof(float)*B*N*S*PD, 128);
	KN = (float*)aligned_malloc(sizeof(float)*B*N*S*PD, 128);//
 	VN = (float*)aligned_malloc(sizeof(float)*B*N*S*PD, 128);
	VNT = (float*)aligned_malloc(sizeof(float)*S*PD, 128);

	
#define REPEAT_N 5
	//开始计算
	//01 计算q,k,v
	TIME_T st, ed;
	MARK_TIME(st);
	for(int b = 0; b < B; b ++)
    {	
		/*	
        _local_gemm_rcr(x+b*S*D, D, w, D, Q+b*S*D, D, S, D, D);
        _local_gemm_rcr(x+b*S*D, D, w+D*D, D, K+b*S*D, D, S, D, D);
        _local_gemm_rcr(x+b*S*D, D, w+2*D*D, D, V+b*S*D, D, S, D, D);
		*/
		dt->b = b;
		athread_spawn(gemm_rcr1,dt);
		athread_join();
    }
	MARK_TIME(ed);
	 LOG("01 time : %.3f ms", DIFF_TIME(st, ed));
	//02分头
	MARK_TIME(st);
	//_local_trans_head(Q, QN, B, S, D, N);
    _local_trans_head(K, KN, B, S, D, N);
    _local_trans_head(V, VN, B, S, D, N);
	MARK_TIME(ed);
	 LOG("02 time : %.3f ms", DIFF_TIME(st, ed));
	//printf("time 2 is %ld\n",ed-st);
	//printf("2 is ok\n");
	//03 计算qk
	MARK_TIME(st);
#define NI(b,n,s,pd) ((((b)*N+n)*S+s)*PD+pd)
#define QKI(b,n,sh,sl) ((((b)*N+n)*S+sh)*S+sl)
	// QK = Q*KT
	for(int b = 0; b < B; b ++)
		
		{
			dt->b=b;
			
			//_local_gemm_rcr(QN+NI(b,n,0,0), PD, KN+NI(b,n,0,0), PD, QK+QKI(b,n,0,0), S, S, S, PD);
		athread_spawn(gemm_rcr2,dt);
		athread_join();
		}
	MARK_TIME(ed);
	 LOG("03 time : %.3f ms", DIFF_TIME(st, ed));
	//printf("time 3 is %ld\n",ed-st);
	//printf("3 is ok\n");

	//04归一化
	MARK_TIME(st);
	double norm = sqrt(PD*1.0);
	if(B*N*S*S<64){
	//printf("2.... is ok,%f\n",QK[45]);
	for(int i = 0; i < B*N*S*S; i ++)
	{
		QK[i] /= norm;
		//printf("%d is ok\n",i);
	}	
	}else{
		athread_spawn(sa_norm1,dt);
		athread_join();
	}
	
	if(B*N*S<64){
		for(int b = 0; b < B; b ++)
			for(int n = 0; n < N; n ++)
				for(int s = 0; s < S; s ++)
					_local_norm(QK+QKI(b,n,s,0), S);
	}else{
		athread_spawn(sa_norm2,dt);
		athread_join();
	}
	// reuse Q
	MARK_TIME(ed);
	 LOG("04 time : %.3f ms", DIFF_TIME(st, ed));
	//printf("time 4 is %ld\n",ed-st);
	memset(QN, 0, sizeof(float)*B*S*D);
	//printf("4 is ok\n");
	//05 计算qkv
	//MARK_TIME(st);
	int count=0;
	MARK_TIME(st);
	for(int b = 0; b < B; b ++)
	{
		for(int n = 0; n < N; n ++)
		{	
			for(int i=0;i<PD;i++){
				for(int j=0;j<S;j++){
					*(KN+NI(b,n,0,0)+i*S+j)=*(VN+NI(b,n,0,0)+j*PD+i);
				}
			}
		}	
	}
	for(int b = 0; b < B; b ++)
		{
			
			dt->b=b;
		athread_spawn(gemm_rrr,dt);
		athread_join();
	}
		
	MARK_TIME(ed);
	LOG("05 time : %.3f ms", DIFF_TIME(st, ed));
	//printf("time 5 is %ld\n",ed-st);
	//printf("5 is ok\n");
    //trans back
	_local_trans_head_back(QN, y, B, S, D, N);
    
	aligned_free(QN);
	aligned_free(KN);
	aligned_free(VN);

	/*
    athread_spawn(par_multihead_attn, arg); // spawn
    athread_join(); // wait for all slave threads finished
	*/
    return 0;
}

