#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
__global__   void SmallWindowBurst(float *dx, int n, int k, float *dxbar, float *dCandMeans, float *dPreCandMeans, float *sum, int *startm1){
   int tid=threadIdx.y*blockDim.x+threadIdx.x;
   int me =blockIdx.x*blockDim.x*blockDim.y+tid;//me [0 to n-k+1)
   int winsize=me+k;//[k to n+1)
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
   startm1[me]=0;
   if(me==0){
     for(int i=n-k+1; i<n; i++){
       sx[i]=dx[i];
     }
   }
   
   
   __syncthreads();

   dCandMeans[me]=sum[me]/winsize;
   dPreCandMeans[me]=sum[me]/winsize;
   //printf("av=%f, %d\n", dCandMeans[me], winsize);
   if(winsize==n){
      dxbar[winsize]=dCandMeans[me];
      return;
   }
   //now find rest of means, rolling window
   dxbar[me]=dCandMeans[me];
   for(; startm1[me]<(n-winsize); startm1[me]++){
      dPreCandMeans[me]=dCandMeans[me];
      dCandMeans[me]=dPreCandMeans[me]+((sx[winsize+startm1[me]]-sx[startm1[me]])/winsize);
      
     // printf("start, maxCand, n-winsize=%d, %f, %d\n", startm1[me]+1, dCandMeans[me], n-winsize );
      if(dCandMeans[me]>dxbar[me]){
         dxbar[me]=dCandMeans[me];
      }
   }
   //printf("%d\n",maxCand);
   
   printf("dxbar[winzie], winsize=%f, %d\n", dxbar[me], winsize);
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
    cudaMalloc ((void **)&dxbar, sizeof(float)*(n-k+1)*(n-k+1));
    cudaMalloc (( void **)&dout, nblk*sizeof(float));
    cudaMemcpy(dx,x,asize ,cudaMemcpyHostToDevice);
    cudaMemcpy(dbigmax, bigmax, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dout, out, sizeof(float)*(nblk), cudaMemcpyHostToDevice);

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
       nblk=(n-k+1)/256+1;
       cudaMalloc ((void **)&dxbar, sizeof(float)*(n-k+1));
       cudaMemcpy(dxbar,xbar,sizeof(float)*(n-k+1) ,cudaMemcpyHostToDevice);
       
       float* CandMeans;
       CandMeans=(float*) malloc(sizeof(float)*(n-k+1));
       float* dCandMeans;
       cudaMalloc ((void **)&dCandMeans, sizeof(float)*(n-k+1));
       cudaMemcpy(dCandMeans, CandMeans, sizeof(float)*(n-k+1), cudaMemcpyHostToDevice);
       
       float* PreCandMeans;
       PreCandMeans=(float*) malloc(sizeof(float)*(n-k+1));
       float* dPreCandMeans;
       cudaMalloc ((void **)&dPreCandMeans, sizeof(float)*(n-k+1));
       cudaMemcpy(dPreCandMeans, PreCandMeans, sizeof(float)*(n-k+1), cudaMemcpyHostToDevice);
       
       float* Sums;
       Sums=(float*) malloc(sizeof(float)*(n-k+1));
       for(int i=0; i<k;i++){
          Sums[0]+=x[i];
       }
       for(int i=1; i<n-k+1;i++){
          Sums[i]=Sums[i-1]+x[i+k-1];
       }
       float* dSums;
       cudaMalloc ((void **)&dSums, sizeof(float)*(n-k+1));
       cudaMemcpy(dSums, Sums, sizeof(float)*(n-k+1), cudaMemcpyHostToDevice);

       int* Startm1;
       int* dStartm1;
       Startm1=(int*) malloc(sizeof(int)*(n-k+1));
       cudaMalloc ((void **)&dStartm1, sizeof(int)*(n-k+1));
       cudaMemcpy(dStartm1, Startm1, sizeof(int)*(n-k+1), cudaMemcpyHostToDevice);
       
       
       SmallWindowBurst<<<dimGrid, dimBlock, nblk*n>>>(dx, n, k, dxbar, dCandMeans, dPreCandMeans, dSums, dStartm1);
       cudaFree(dSums);
       cudaFree(dPreCandMeans);
       cudaFree(dCandMeans);
       cudaFree(dStartm1);
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
  int n=100;
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
