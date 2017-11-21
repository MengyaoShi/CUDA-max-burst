#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
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
   if (perstart<=n-k && perlen>=k && perlen<=maxWinSize){
      perend=perstart+perlen-1;
      int i; float tot=0;
      for(i=perstart;i<=perend;i++) tot+=dx[i];
      dxbar[indx]=tot/(perend-perstart+1);
   }
   else{
      printf("mean, indx=%f, %d\n", dxbar[indx], indx);
      return;
   }
   __syncthreads();
   printf("mean,indx=%f, %d\n", dxbar[indx], indx);
}

__global__ void reduce(float *g_idata, float *g_odata){
   printf("hwhew");
   extern __shared__ float sdata[];
   int tid=threadIdx.y*blockDim.x+threadIdx.x;
   unsigned int i=blockIdx.x*blockDim.x*blockDim.y+tid;
   sdata[tid]=g_idata[i];
   __syncthreads();

   for(unsigned int s=1; s<blockDim.x*blockDim.y; s*=2){
      int index=2*s*tid;
      printf("print something");
      if(index<blockDim.x*blockDim.y){
         if(g_idata[tid]<g_idata[index+s]){
            g_idata[tid]=g_idata[index+s];
         }
      }
      __syncthreads();

   }

   if(tid==0) g_odata[blockIdx.x]=g_idata[0];
   printf("in reduce, %f,\n", g_odata[blockIdx.x]);
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
    int nblk=1;//Number of blocks
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
    cudaMemcpy(dxbar,xbar,sizeof(float)*(n-k+1)*(n-k+1) ,cudaMemcpyHostToDevice);
    cudaMemcpy(dout, out, sizeof(float)*(nblk), cudaMemcpyHostToDevice);

    // set up parameters for threads structure
    dim3 dimGrid(nblk,1); // n blocks
    dim3 dimBlock(n-k+1, n-k+1,1);
    // invoke the ker
    // make winsize
    if(n>2*k){
       maxWinSize=2*k;
    }

    burst<<<dimGrid,dimBlock>>>(dx,n,k,dxbar, maxWinSize);
    cudaThreadSynchronize();
    cudaMemcpy(xbar, dxbar, sizeof(float)*(n-k+1)*(n-k+1), cudaMemcpyDeviceToHost);
    int tmp=0;
    for(tmp=0; tmp<(n-k+1)*(n-k+1); tmp++){
       printf("after copy from GPU to CPU, mean, indx  are %f, %d\n", xbar[tmp], tmp);
    }
    cudaMemcpy(dxbar,xbar,sizeof(float)*(n-k+1)*(n-k+1) ,cudaMemcpyHostToDevice);
    //SomeReduce function
    reduce<<<dimGrid, dimBlock>>>(dxbar, dout);
    // copy row vector from device to host
    cudaMemcpy(xbar, dxbar, sizeof(float)*(n-k+1)*(n-k+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(bigmax,dbigmax, sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(out, dout, sizeof(float)*nblk, cudaMemcpyDeviceToHost);
    printf("bigmax is%f\n", xbar[0]);
    cudaFree(dxbar);
    cudaFree(dout);
    cudaFree (dbigmax);
    cudaFree (dx);

}
int main(int arc, char **argv){
  float *x;
  int n=10;
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
