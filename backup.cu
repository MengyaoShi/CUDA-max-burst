#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
__global__   void SmallWindowBurst(float *dx, int n, int k, float *dxbar){
   int tid=threadIdx.y*blockDim.x+threadIdx.x;
   int me =blockIdx.x*blockDim.x*blockDim.y+tid;//me [0 to n-k+1)
   int winsize=me+k;//[k to n+1)
   float sum=0.0;
   float dCandMean=0.0;
   float dPreCandMean=0.0;
   if(winsize>n){
      return;
   }
   //dxbar is of dimension, n-k+1
   if(n<2*k){
      return;
   }
   extern __shared__ float sx[];//of dimension, n
   //copy dx to sx
   sx[me]=dx[me];
   int startm1=0;
   if(me==0){
     for(int i=n-k+1; i<n; i++){
       sx[i]=dx[i];
     }
   }
   __syncthreads();

   for(int i=0;i<winsize;i++){
      sum+=sx[i];
   }
   dxbar[me]=sum/winsize;
   //printf("av=%f, %d\n", dxbar[me], winsize);

   dCandMean=sum/winsize;
   dPreCandMean=sum/winsize;
   if(winsize==n){
      dxbar[me]=dCandMean;
      return;
   }
   //now find rest of means, rolling window
   dxbar[me]=dCandMean;
   for(; startm1<(n-winsize); startm1++){
      dPreCandMean=dCandMean;
      dCandMean=dPreCandMean+((sx[winsize+startm1]-sx[startm1])/winsize);
      if(winsize==3){ 
         printf("start, maxCand, n-winsize=%d, %f, %d\n", startm1+1, dCandMean, n-winsize );
      }
      if(dCandMean>dxbar[me]){
         dxbar[me]=dCandMean;
      }
   }
   //printf("%d\n",maxCand);
   
   //printf("dxbar[winzie], winsize=%f, %d\n", dxbar[me], winsize);
}


__global__   void burst(float *dx, int n, int k, float *dxbar, int maxWinSize) {
   int tid=threadIdx.y*blockDim.x+threadIdx.x;
   int me=blockIdx.x*blockDim.x*blockDim.y+tid;
   int width=n-k+1;
   int x=me%width;
   int y=me/width;
   int perstart=x;//start
   int perend;
   int indx=0;
   extern __shared__ float sx[];
   int perlen=y+k;//length of window, or window size. Notice if minimum windowSize k is smaller than n/2 ,we only need maximum windowSize to be 2k.
   //each thread copy one number to shared memory, notice we have more threads than numbers/
   indx=perstart*(n-k+1)+perlen-k;
   dxbar[indx]=-1000.0;
   if(me<n){
      sx[me]=dx[me];
   }
   __syncthreads();
   if(maxWinSize>n-perstart){
     maxWinSize=n-perstart;
   } 
   if (perstart<=n-k && perlen>=k && perlen<=maxWinSize && n<2*k){
      perend=perstart+perlen-1;
      int i; float tot=0;
      for(i=perstart;i<=perend;i++) tot+=sx[i];
      dxbar[indx]=tot/(perend-perstart+1);
   }
   else{
      return;
   }
   __syncthreads();
   //printf("mean,indx=%f, %d\n", dxbar[indx], indx);
}

__global__ void reduce(float *g_idata, float *g_odata){
   extern __shared__ float sdata[];
   int tid=threadIdx.y*blockDim.x+threadIdx.x;
   unsigned int i=blockIdx.x*blockDim.x*blockDim.y+tid;
   sdata[tid]=g_idata[i];
   __syncthreads();
   //printf("sdata[tid],tid=%f, %d\n", sdata[tid], tid);
   for(unsigned int s=1; s<blockDim.x*blockDim.y; s*=2){
      int index=2*s*tid;
      if(index<blockDim.x*blockDim.y){
         if(sdata[index]<sdata[index+s]){
            sdata[index]=sdata[index+s];
         }
      }
      __syncthreads();

   }

   if(tid==0) {g_odata[blockIdx.x]=sdata[0];
   printf("in reduce, %f,\n", g_odata[blockIdx.x]);}
}

// things need to fix probably: bigmax allocate one int; passing n and k and bigmax to cuda function
void maxburst(float *x, int n, int k, int *startend, float *bigmax){
    float *dx; //device x
    float *dbigmax; //device bigmax
    int asize = n*sizeof(float);
    float *out;//each block has an output max mean answer.
    float *dout; //on device, out.

    float* xbar; //Means for every possiblle start position, and window size.
    float* dxbar;
    int nblk=(n-k+1)*(n-k+1)/256+1;//Number of blocks
    int maxWinSize=n;
    // copy host matrix to device matrix

    xbar=(float *) malloc(sizeof(float)*(n-k+1)*(n-k+1));
    out=(float *) malloc(sizeof(float)*nblk);
    // allocate space for device matrix
    cudaMalloc ((void **)&dx,asize);
    cudaMalloc (( void **)&dbigmax , sizeof(float) );
    cudaMalloc (( void **)&dout, nblk*sizeof(float));
    cudaMemcpy(dbigmax, bigmax, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dout, out, sizeof(float)*(nblk), cudaMemcpyHostToDevice);
    cudaMemcpy(dx,x,sizeof(float)*n, cudaMemcpyHostToDevice);

    // invoke the ker
    // make winsize
    if(n<2*k){
       maxWinSize=2*k;
       nblk=(n-k+1)*(n-k+1)/256+1;
    }
    else{
       nblk=(n-k+1)/256+1;
    }
    dim3 dimGrid(nblk,1); // n blocks
    dim3 dimBlock(16, 16,1);
    if(n<2*k){
       cudaMalloc ((void **)&dxbar, sizeof(float)*(n-k+1)*(n-k+1));
       cudaMemcpy(dxbar,xbar,sizeof(float)*(n-k+1)*(n-k+1) ,cudaMemcpyHostToDevice);
       burst<<<dimGrid,dimBlock, n*nblk>>>(dx,n,k,dxbar, maxWinSize);
    }
    else{
       cudaMalloc ((void **)&dxbar, sizeof(float)*(n-k+1));
       cudaMemcpy(dxbar,xbar,sizeof(float)*(n-k+1) ,cudaMemcpyHostToDevice);
       
       
       SmallWindowBurst<<<dimGrid, dimBlock, nblk*n>>>(dx, n, k, dxbar);
    }
    //If the wind size is smaller than n/2, we are goint to use first approach. n-k+1, in second senario, we have (n-k+1) **2
    
    
    cudaThreadSynchronize();
    cudaMemcpy(xbar, dxbar, sizeof(float)*(n-k+1)*(n-k+1), cudaMemcpyDeviceToHost);
    int tmp=0;
    for(tmp=0; tmp<(n-k+1)*(n-k+1); tmp++){
       //printf("after copy from GPU to CPU, mean, indx  are %f, %d\n", xbar[tmp], tmp);
    }
    cudaMemcpy(dxbar,xbar,sizeof(float)*(n-k+1)*(n-k+1) ,cudaMemcpyHostToDevice);
    //SomeReduce function
    reduce<<<dimGrid, dimBlock, (n-k+1)*(n-k+1)>>>(dxbar, dout);
    // copy row vector from device to host
    //cudaMemcpy(xbar, dxbar, sizeof(float)*(n-k+1)*(n-k+1), cudaMemcpyDeviceToHost);
    //cudaMemcpy(bigmax,dbigmax, sizeof(float),cudaMemcpyDeviceToHost);
    //cudaMemcpy(out, dout, sizeof(float)*nblk, cudaMemcpyDeviceToHost);
    //printf("bigmax is%f\n", xbar[0]);
    cudaFree(dxbar);
    cudaFree(dout);
    cudaFree (dbigmax);
    cudaFree (dx);

}
int main(int arc, char **argv){
  float *x;
  int n=80;
  int k=3;
  int *startend;
  float *bigmax;
  bigmax=(float*) malloc(sizeof(float));
  startend=(int*) malloc(sizeof(int)*2);
  x=(float*) malloc(sizeof(float)*n);
  int i;
  for(i=0; i<n; i++){
     x[i]=i*1.0;
  }
  bigmax[0]=0;
  maxburst(x, n, k, startend, bigmax);
} 
