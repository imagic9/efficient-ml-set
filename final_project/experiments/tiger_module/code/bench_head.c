/* Marginal latency of a "new-target" head on the Pi, on top of the existing
 * backbone forward pass. The backbone already produces a 1280-d pooled embedding;
 * adding a target animal is: (A1) one cosine = dot(1280) + L2 norm; (A2/A3) one
 * linear row = dot(1280)+bias. A full head is [N x 1280]. We time these directly. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define D 1280
static volatile double sink = 0.0;

static double now_s(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9;
}
static float dot(const float*a,const float*b){
    float s=0.f; for(int i=0;i<D;i++) s+=a[i]*b[i]; return s;
}

int main(void){
    srand(0);
    float x[D], proto[D];
    for(int i=0;i<D;i++){ x[i]=(float)rand()/RAND_MAX; proto[i]=(float)rand()/RAND_MAX; }
    // L2 normalise proto once (prototype is precomputed & stored normalised)
    float n=sqrtf(dot(proto,proto)); for(int i=0;i<D;i++) proto[i]/=n;

    int Ns[]={1,16,64}; // 1 new target, full 16-class head, 64 targets catalog
    // weight matrices
    for(int k=0;k<3;k++){
        int N=Ns[k];
        float* W=malloc((size_t)N*D*sizeof(float));
        for(long i=0;i<(long)N*D;i++) W[i]=(float)rand()/RAND_MAX - 0.5f;
        long REPS = 200000;
        // warmup
        for(long r=0;r<2000;r++){ float acc=0; for(int j=0;j<N;j++) acc+=dot(&W[(size_t)j*D],x); sink+=acc; }
        double t0=now_s();
        for(long r=0;r<REPS;r++){
            float acc=0; for(int j=0;j<N;j++) acc+=dot(&W[(size_t)j*D],x);
            sink+=acc;
        }
        double dt=now_s()-t0;
        printf("linear head  N=%2d targets:  %.4f us/frame   (%.1f M dot-ops/s)\n",
               N, dt/REPS*1e6, (double)N*REPS/dt/1e6);
        free(W);
    }
    // prototype path: cosine = dot + normalise x + divide
    {
        long REPS=200000;
        for(long r=0;r<2000;r++){ float nx=sqrtf(dot(x,x)); float c=dot(x,proto)/nx; sink+=c; }
        double t0=now_s();
        for(long r=0;r<REPS;r++){
            float nx=sqrtf(dot(x,x)); float c=dot(x,proto)/nx; sink+=c;
        }
        double dt=now_s()-t0;
        printf("prototype cosine (1 target): %.4f us/frame\n", dt/REPS*1e6);
    }
    printf("sink=%.3f\n",(double)sink);
    return 0;
}
