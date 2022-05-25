#include <slave.h>
#include <math.h>
#include <simd.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include "simd.h"

#include "args.h"


#define A AX
#define B BX

static inline unsigned long rpcc() {
	unsigned long time;
	
    __asm__ __volatile__ ("rcsr %0,4":"=r"(time));
	//#define rpcc(time) asm volatile("rcsr %0,4":"=r"(time))
	return time;
}


__thread_local int id;
__thread_local volatile unsigned long get_reply, put_reply,put_reply2[2];
__thread_local float AX[4096];//16KB
__thread_local float BX[4096];//16kB

__thread_local float C;
__thread_local float CX[128];


typedef struct Dt
{
    int b; // batch
    int S; // sequence length
    int D; // vector size
    int n; //head number
	int N;//heads
    int B;
}Dts, *Dt_a;

extern const float* x;
extern const float* w;
extern float* Q;
extern float* K;
extern float* V;
extern float* QK;
extern float* QN;
extern float* KN;
extern float* VN;
extern float* VNT;

#define NI(b,n,s,pd) ((((b)*N+n)*S+s)*PD+pd)
#define QKI(b,n,sh,sl) ((((b)*N+n)*S+sh)*S+sl)

void par_multihead_attn(Args_t arg)
{
	const int id = athread_get_id(-1);
	if(id == 0)
		printf("passed\n");
}

void init(){
    C=0.0;
    for(int i=0;i<64;i++){
        CX[i]=0.0;
    }
}
void init2(){
    for(int i=64;i<128;i++){
        CX[i]=0.0;
    }
}

// X*W 
void gemm_rcr1(Dt_a dt)
{

    long st,ed;
    long sum = 8327127;//测试得到的主要部分
    if(id == 0){
        //printf("sum = %ld\n",sum);
    }
	int b,S,D;
	int o;
	b = dt->b;
	D =dt->D;
	S =dt->S;
    init();
    int sp=0,sq=0;
    int dp=0,dq=0;
    id = athread_get_id(-1);
    int countx=0,countw=0;
    //先分X
    sp = S%2;
    for(countx=0;countx<S;countx+=2){
        get_reply = 0;
        //X:初地址+countx+id
        athread_get(PE_MODE, x+b*S*D+D*countx, &AX[0], D*2*4, &get_reply, 0, 0, 0);
        while(get_reply!=1);
        dp = D%64;//不能全给的情况
        dq = D/64;//dq!=0,1


      
        //wq
        for(countw=0;countw<dq;countw+=2){
            get_reply = 0;   
            //重点 id*dq*D +countw*2*D
            athread_get(PE_MODE, w+id*dq*D +countw*D, &BX[0], D*2*4, &get_reply, 0, 0, 0);
            while (get_reply != 1);
            //矩阵乘 simd
            for(o = 0;o< D; o++){
                CX[countw]+= AX[o]*BX[o];
                CX[countw+1]+= AX[o]*BX[o+D];
                CX[countw+dq]+= AX[o+D]*BX[o];
                CX[countw+dq+1]+= AX[o+D]*BX[o+D];
            }
        }
        put_reply = 0;
	    athread_put(PE_MODE,&CX[0], Q+b*S*D+D*countx+dq*id, dq*4, &put_reply, 0, 0);
        athread_put(PE_MODE,&CX[dq], Q+b*S*D+D*(countx+1)+dq*id, dq*4, &put_reply, 0, 0);
	    while (put_reply != 2);
        init();//注意
        
        for(int j=0;j<dp;j++){
            if(j==id){
                get_reply = 0;
                athread_get(PE_MODE, w+64*D*(countw-1)+D*id, &B[0], D*4, &get_reply, 0, 0, 0);
                while(get_reply!=1);
                for(o = 0;o< D; o++){
                CX[0]+= AX[o]*B[o];
                CX[1]+=AX[o+D]*B[o];
                }
            put_reply = 0;
	        athread_put(PE_MODE,&CX[0], Q+b*S*D+D*countx+64*(countw-1)+id, 4, &put_reply, 0, 0);
            athread_put(PE_MODE,&CX[1], Q+b*S*D+D*(countx+1)+64*(countw-1)+id, 4, &put_reply, 0, 0);
	        while (put_reply != 2);
            CX[0]=0.0;
            CX[1]=0.0;
            }
        }


        //wk
       for(countw=0;countw<dq;countw+=2){
            get_reply = 0;   
            //重点 id*dq*D +countw*2*D
            athread_get(PE_MODE, w+D*D+id*dq*D +countw*D, &BX[0], D*2*4, &get_reply, 0, 0, 0);
            while (get_reply != 1);
            //矩阵乘 simd
            for(o = 0;o< D; o++){
                CX[countw]+= AX[o]*BX[o];
                CX[countw+1]+= AX[o]*BX[o+D];
                CX[countw+dq]+= AX[o+D]*BX[o];
                CX[countw+dq+1]+= AX[o+D]*BX[o+D];
            }
        }
        put_reply = 0;
	    athread_put(PE_MODE,&CX[0], K+b*S*D+D*countx+dq*id, dq*4, &put_reply, 0, 0);
        athread_put(PE_MODE,&CX[dq], K+b*S*D+D*(countx+1)+dq*id, dq*4, &put_reply, 0, 0);
	    while (put_reply != 2);
        init();//注意
        for(int j=0;j<dp;j++){
            if(j==id){
                get_reply = 0;
                athread_get(PE_MODE, w+D*D+64*D*(countw-1)+D*id, &B[0], D*4, &get_reply, 0, 0, 0);
                while(get_reply!=1);
                for(o = 0;o< D; o++){
                CX[0]+= AX[o]*B[o];
                CX[1]+=AX[o+D]*B[o];
                }
            put_reply = 0;
	        athread_put(PE_MODE,&CX[0], K+b*S*D+D*countx+64*(countw-1)+id, 4, &put_reply, 0, 0);
            athread_put(PE_MODE,&CX[1], K+b*S*D+D*(countx+1)+64*(countw-1)+id, 4, &put_reply, 0, 0);
	        while (put_reply != 2);
            CX[0]=0.0;
            CX[1]=0.0;
            }
        }
        
        //wv
         for(countw=0;countw<dq;countw+=2){
            get_reply = 0;   
            //重点 id*dq*D +countw*2*D
            athread_get(PE_MODE, w+D*D*2+id*dq*D +countw*D, &BX[0], D*2*4, &get_reply, 0, 0, 0);
            while (get_reply != 1);
            //矩阵乘 simd
            for(o = 0;o< D; o++){
                CX[countw]+= AX[o]*BX[o];
                CX[countw+1]+= AX[o]*BX[o+D];
                CX[countw+dq]+= AX[o+D]*BX[o];
                CX[countw+dq+1]+= AX[o+D]*BX[o+D];
            }
        }
        put_reply = 0;
	    athread_put(PE_MODE,&CX[0], V+b*S*D+D*countx+dq*id, dq*4, &put_reply, 0, 0);
        athread_put(PE_MODE,&CX[dq], V+b*S*D+D*(countx+1)+dq*id, dq*4, &put_reply, 0, 0);
	    while (put_reply != 2);
        init();//注意
        for(int j=0;j<dp;j++){
            if(j==id){
                get_reply = 0;
                athread_get(PE_MODE, w+D*D*2+64*D*(countw-1)+D*id, &B[0], D*4, &get_reply, 0, 0, 0);
                while(get_reply!=1);
                for(o = 0;o< D; o++){
                CX[0]+= AX[o]*B[o];
                CX[1]+=AX[o+D]*B[o];
                }
            put_reply = 0;
	        athread_put(PE_MODE,&CX[0], V+b*S*D+D*countx+64*(countw-1)+id, 4, &put_reply, 0, 0);
            athread_put(PE_MODE,&CX[1], V+b*S*D+D*(countx+1)+64*(countw-1)+id, 4, &put_reply, 0, 0);
	        while (put_reply != 2);
            CX[0]=0.0;
            CX[1]=0.0;
            }
        }
    }
    if(sp!=0){
        countx=S-1;   
        get_reply = 0;
        //X:初地址+countx+id
        athread_get(PE_MODE, x+b*S*D+D*countx, &A[0], D*4, &get_reply, 0, 0, 0);
        while(get_reply!=1);
		
        //wq
        dp = D%64;//不能全给的情况
        dq = D/64;
        for(countw=0;countw<dq;countw++){
            //通信Wq
			if(id==1){
			//printf("03 is ok!,%d\n",countw);
		}
            get_reply = 0;
            athread_get(PE_MODE, w+64*D*countw+D*id, &B[0], D*4, &get_reply, 0, 0, 0);//问题
            while(get_reply!=1);
            //矩阵乘
			if(id==1){
			//printf("04 is ok!\n");
		}
            for(o = 0;o< D; o++){
                C+= A[o]*B[o];
            }
            put_reply = 0;
	        athread_put(PE_MODE,&C, Q+b*S*D+D*countx+64*countw+id, 4, &put_reply, 0, 0);
	        while (put_reply != 1);
            C=0.0;
          
        }
        for(int j=0;j<dp;j++){
            if(j==id){
                get_reply = 0;
                athread_get(PE_MODE, w+64*D*countw+D*id, &B[0], D*4, &get_reply, 0, 0, 0);
                while(get_reply!=1);
                for(o = 0;o< D; o++){
                C+= A[o]*B[o];
            }
            put_reply = 0;
	        athread_put(PE_MODE, &C,Q+b*S*D+D*countx+64*countw+id, 4, &put_reply, 0, 0);
	        while (put_reply != 1);
			C=0.0;
            }
        }


        //wk
        for(countw=0;countw<dq;countw++){
            //通信Wq
            get_reply = 0;
            athread_get(PE_MODE, w+D*D+64*D*countw+D*id, &B[0], D*4, &get_reply, 0, 0, 0);
            while(get_reply!=1);
            //矩阵乘
           for(o = 0;o< D; o++){
                C+= A[o]*B[o];
            }
            put_reply = 0;
             athread_put(PE_MODE,&C, K+b*S*D+D*countx+64*countw+id, 4, &put_reply, 0, 0);
	        while (put_reply != 1);
            C=0.0;
        }
       for(int j=0;j<dp;j++){
            if(j==id){
                get_reply = 0;
                athread_get(PE_MODE, w+D*D+64*D*countw+D*id, &B[0], D*4, &get_reply, 0, 0, 0);
                while(get_reply!=1);
                for(o = 0;o< D; o++){
                C+= A[o]*B[o];
            }
            put_reply = 0;
	        athread_put(PE_MODE,&C,K+b*S*D+D*countx+64*countw+id, 4, &put_reply, 0, 0);
	        while (put_reply != 1);
			C=0.0;
            }
        }
        
        //wv
        for(countw=0;countw<dq;countw++){
            //通信Wq
            get_reply = 0;
            athread_get(PE_MODE, w+D*D*2+64*D*countw+D*id, &B[0], D*4, &get_reply, 0, 0, 0);
            while(get_reply!=1);
            //矩阵乘
            for(o = 0;o< D; o++){
                C+= A[o]*B[o];
            }
            put_reply = 0;
             athread_put(PE_MODE,&C, V+b*S*D+D*countx+64*countw+id, 4, &put_reply, 0, 0);
	        while (put_reply != 1);
            C=0.0;
        }
       for(int j=0;j<dp;j++){
            if(j==id){
                get_reply = 0;
                athread_get(PE_MODE, w+D*D*2+64*D*countw+D*id, &B[0], D*4, &get_reply, 0, 0, 0);
                while(get_reply!=1);
                for(o = 0;o< D; o++){
                C+= A[o]*B[o];
            }
            put_reply = 0;
	        athread_put(PE_MODE,&C,V+b*S*D+D*countx+64*countw+id, 4, &put_reply, 0, 0);
	        while (put_reply != 1);
			C=0.0;
            }
        }
       
    }
    
}

void gemm_rcr2(Dt_a dt)
{
    int b,S,D,n,N;
	int o;
	b = dt->b;
	D =dt->D;
	S =dt->S;
    N = dt->N;
    int PD=D/N;

    init();
    init2();
    float tag1=0.0;
    float tag2=0.0;
    int sp=0,sq=0;
    int dp=0,dq=0;
    long st,ed;
    id = athread_get_id(-1);
    int countx=0,countw=0;
    //先分qn
    int flag= 1;//双缓存
    put_reply2[1] = 2;
    if(S==1){
        countx = 1;
    }
    for(countx=0;countx<S;countx+=2){
        get_reply = 0;
        //qn:初地址+countx
        athread_get(PE_MODE, Q+b*S*D+D*countx, &AX[0], D*2*4, &get_reply, 0, 0, 0);
        while(get_reply!=1);
		
        //KN
        dp = S%64;//不能全给的情况
        dq = S/64;
        
        for(countw=0;countw<dq;countw++){
            //通信Wq
            st = rpcc();
            get_reply = 0;
            athread_get(PE_MODE, K+b*S*D+64*D*countw+D*id, &BX[0], D*4, &get_reply, 0, 0, 0);
            while(get_reply!=1);
            ed = rpcc();
            if(id==0&&countx==0&&countw==0){
               // printf("t1 time is %ld\n",ed-st);
            }

            flag = (flag+1)%2;//0->63 or 64->123

            st = rpcc();
            //矩阵乘
            for(n=0;n<N;n++)
            {
                for(o = 0;o< PD; o++){
                    CX[n+flag*64] +=AX[o+n*PD]*BX[o+n*PD];
                    CX[n+flag*64+N] +=AX[o+n*PD+D]*BX[o+n*PD];
                }
            }
             ed = rpcc();
            if(id==0&&countx==0&&countw==0){
                //printf("t2 time is %ld\n",ed-st);
            }
            st = rpcc();
            put_reply2[flag] = 0;
            athread_put(PE_MODE, &CX[0+flag*64], QK+QKI(b,0,0,0)+S*countx+64*countw+id, N*4, &put_reply2[flag], S*S*4-4, 4);
            athread_put(PE_MODE, &CX[0+flag*64+N], QK+QKI(b,0,0,0)+S*(countx+1)+64*countw+id, N*4, &put_reply2[flag], S*S*4-4, 4);
	        while (put_reply2[(flag+1)%2] != 2);
            ed = rpcc();
            if(id==0&&countx==0&&countw==0){
               // printf("t3 time is %ld\n",ed-st);
            }
            if(!flag){
                init2();
            }else{
                init();
            }
            
        }
        
        if(countw==0){
            countw=1; 
        }
        for(int j=0;j<dp;j++){
            if(j==id){
                get_reply = 0;
                athread_get(PE_MODE, K+NI(b,0,0,0)+64*D*(countw-1)+D*id, &B[0], D*4, &get_reply, 0, 0, 0);
                while(get_reply!=1);
                
               put_reply = 0;
                for(n=0;n<N;n++)
                {
                    for(o = 0;o< PD; o++){
                        CX[n] +=AX[o+n*PD]*BX[o+n*PD];
                        CX[n+N] +=AX[o+n*PD+D]*BX[o+n*PD];
                    }
                athread_put(PE_MODE,&CX[n], QK+QKI(b,n,0,0)+S*countx+64*(countw-1)+id, 4, &put_reply, 0, 0);
                athread_put(PE_MODE,&CX[n+N], QK+QKI(b,n,0,0)+S*(countx+1)+64*(countw-1)+id, 4, &put_reply, 0, 0);
                }
	            while (put_reply != 2*N);
                init();
        }
    }
    if(S%2==1){
            if(S==1){
                countx = 0;
            }else{
                countx = S-1;
            }
            get_reply = 0;
            athread_get(PE_MODE, K+b*S*D+64*D*countw+D*id, &BX[0], D*4, &get_reply, 0, 0, 0);
            while(get_reply!=1);
            //矩阵乘
            put_reply = 0;
            
            for(n=0;n<N;n++)
            {
                for(o = 0;o< PD; o++){
                    CX[n] +=AX[o+n*PD]*BX[o+n*PD];
                }
                athread_put(PE_MODE,&CX[n], QK+QKI(b,n,0,0)+S*countx+64*countw+id, 4, &put_reply, 0, 0);
            }
	        while (put_reply != N);
            init();
            if(countw==0){
            countw=1; 
            }
            for(int j=0;j<dp;j++){
            if(j==id){
                get_reply = 0;
                athread_get(PE_MODE, K+NI(b,0,0,0)+64*D*(countw-1)+D*id, &B[0], D*4, &get_reply, 0, 0, 0);
                while(get_reply!=1);
                
               put_reply = 0;
                for(n=0;n<N;n++)
                {
                    for(o = 0;o< PD; o++){
                        CX[n] +=AX[o+n*PD]*BX[o+n*PD];
                    }
                athread_put(PE_MODE,&CX[n], QK+QKI(b,n,0,0)+S*countx+64*(countw-1)+id, 4, &put_reply, 0, 0);
                }
	            while (put_reply != N);
                init();
             }
        }

        }




    }
    
}



void gemm_rrr(Dt_a dt)
{   
    int b,S,D,n,N;
	int o;
	b = dt->b;
	D =dt->D;
	S =dt->S;
    n = dt->n;
    N = dt->N;
    int PD=D/N;
    long st,ed;








    for(int n = 0; n < N; n ++){
    init();
    float tag1=0.0;
    float tag2=0.0;
    int sp=0,sq=0;
    int dp=0,dq=0;
    id = athread_get_id(-1);
    int countx=0,countw=0;
    //先分QK
    //for(n=0;n<N;n++)
    {
    for(countx=0;countx<S;countx+=2){
        //通信QK
		
        get_reply = 0;
        //qn:初地址+countx
        athread_get(PE_MODE, QK+QKI(b,n,0,0)+S*countx, &AX[0], S*2*4, &get_reply, 0, 0, 0);
        while(get_reply!=1);
		countw=0;
        //VN
        dp = PD%64;
        dq = PD/64;
        //01
        if(dq!=0){
        for(countw=0;countw<dq;countw++){
            //通信Wq
            get_reply = 0;

            athread_get(PE_MODE, KN+NI(b,n,0,0)+id*dq*S+S*countw, &BX[0], S*4, &get_reply, 0, 0, 0);
            while(get_reply!=1);
            //矩阵乘
	
            for(o = 0;o< S; o++){
                CX[countw]+= AX[o]*BX[o];
                CX[countw+dq]+= AX[o+S]*BX[o];
            }
        }
        put_reply = 0;
	    athread_put(PE_MODE,&CX[0], QN+NI(b,n,0,0)+PD*countx+dq*id, 4*dq, &put_reply, 0, 0);
        athread_put(PE_MODE,&CX[dq], QN+NI(b,n,0,0)+PD*(countx+1)+dq*id, 4*dq, &put_reply, 0, 0);
	    while (put_reply != 2);
        init();
        }
        if(countw==0){
            countw=1; 
        }
        //02
        if(id<dp){
                get_reply = 0;
                athread_get(PE_MODE, KN+NI(b,n,0,0)+64*S*(countw-1)+S*id, &B[0], S*4, &get_reply, 0, 0, 0);
                while(get_reply!=1);
                for(o = 0;o< S; o++){
                CX[0]+= AX[o]*B[o];
                CX[1]+=AX[o+S]*B[o];
            }
            put_reply = 0;
	        athread_put(PE_MODE,&CX[0], QN+NI(b,n,0,0)+PD*countx+64*(countw-1)+id, 4, &put_reply, 0, 0);
            athread_put(PE_MODE,&CX[1], QN+NI(b,n,0,0)+PD*(countx+1)+64*(countw-1)+id, 4, &put_reply, 0, 0);
	        while (put_reply != 2);
            CX[0]=0.0;
            CX[1]=0.0;
            }  
     }
     if(sp!=0){
        countx = S-1;
        get_reply = 0;
        //qn:初地址+countx
        athread_get(PE_MODE, QK+QKI(b,n,0,0)+S*countx, &A[0], S*4, &get_reply, 0, 0, 0);
        while(get_reply!=1);
		
        //VN
        dp = PD%64;//不能全给的情况
        dq = PD/64;
        //01
        for(countw=0;countw<dq;countw++){
            //通信Wq
            get_reply = 0;

            athread_get(PE_MODE, KN+NI(b,n,0,0)+64*S*countw+S*id, &B[0], S*4, &get_reply, 0, 0, 0);//问题
            while(get_reply!=1);
            //矩阵乘
	
            for(o = 0;o< S; o++){
                C+= A[o]*B[o];
            }
            put_reply = 0;
	        athread_put(PE_MODE,&C, QN+NI(b,n,0,0)+PD*countx+64*countw+id, 4, &put_reply, 0, 0);
	        while (put_reply != 1);
            C=0.0;
          
        }
        //02
        for(int j=0;j<dp;j++){
            if(j==id){
                get_reply = 0;
                athread_get(PE_MODE, KN+NI(b,n,0,0)+64*S*countw+S*id, &B[0], S*4, &get_reply, 0, 0, 0);
                while(get_reply!=1);
                for(o = 0;o< S; o++){
                C+= A[o]*B[o];
            }
            put_reply = 0;
	        athread_put(PE_MODE,&C, QN+NI(b,n,0,0)+PD*countx+64*countw+id, 4, &put_reply, 0, 0);
	        while (put_reply != 1);
			C=0.0;
            }
        }
        }
    }
}
}


void sa_norm1(Dt_a dt)
{
    init();
    int N,D,S,B1;
	D =dt->D;
	N =dt->N;
    S = dt->S;
    B1 = dt->B;//命名失误
    int PD=dt->D/dt->N;
    double norm = sqrt(PD*1.0);
    int dq = B1*N*S*S/64;
    int dqs = B1*N*S*S%64;
    id = athread_get_id(-1);
    int dp = dq/1024;
    int dps = dq%1024;
    int borderD;
    int borderU;
    for(int j=0;j<dp;j++){
    //上下界
    borderD = 1024*j;
    borderU = 1024*(j+1);

    //通信
    get_reply = 0;
    athread_get(PE_MODE, QK+dq*id+borderD, &B[0], 1024*4, &get_reply, 0, 0, 0);
    while(get_reply!=1);
    //计算
    for(int i=0;i<1024;i++){
        B[i] /= norm;
       //B[i] = 1;
    }
    //返回
     put_reply = 0;
    athread_put(PE_MODE,&B[0], QK+dq*id+borderD, 1024*4, &put_reply, 0, 0);
    while(put_reply!=1);
    }
    if(dq ==0){
        borderU = 0;
    }
    if(dps!=0 ){
        
        get_reply = 0;
        athread_get(PE_MODE, QK+dq*id+borderU, &B[0], dps*4, &get_reply, 0, 0, 0);
         while(get_reply!=1);
        for(int i=0;i<dps;i++){
             B[i] /= norm;
            
        }
        put_reply = 0;
        athread_put(PE_MODE,&B[0], QK+dq*id+borderU, dps*4, &put_reply, 0, 0);
        while(put_reply!=1);
    }
    if(dqs!=0&&id==0){
        //
        printf("还没改变\n");
    }
}

void sa_norm2(Dt_a dt)
{
    init();
    int N,D,S,B1;
	D =dt->D;
	N =dt->N;
    S = dt->S;
    B1 = dt->B;//命名失误
    int PD=dt->D/dt->N;
    id = athread_get_id(-1);
    int dq = B1*N*S/64;//以行S为单位
    int dqs= B1*N*S%64;
    double sum = 0.0f;
    for(int i=dq*id;i<dq*(id+1);i++){
        sum = 0.0f;
        //通信
        get_reply = 0;
        athread_get(PE_MODE, QK+i*S, &B[0], S*4, &get_reply, 0, 0, 0);
        while(get_reply!=1);
        //计算
        for(int j=0;j<S;j++){
            sum += B[j];
        }
        for(int j=0;j<S;j++){
            B[j] /=sum;
        }
        //传输
        put_reply = 0;
        athread_put(PE_MODE,&B[0], QK+i*S, S*4, &put_reply, 0, 0);
        while(put_reply!=1);

    }
    if(dqs!=0&&id==0){
        //
        printf("还没写-0.0-0.0\n");
    }
}

