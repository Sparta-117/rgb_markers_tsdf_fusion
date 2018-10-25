#include "common/book.h"

__global__ void add(int a,int b,int *c)
{
    *c = a + b;
}

int tsdf()
{
//    cudaDeviceProp  prop;
//    int dev;

//    HANDLE_ERROR( cudaGetDevice( &dev ) );
//    printf( "ID of current CUDA device:  %d\n", dev );

//    memset( &prop, 0, sizeof( cudaDeviceProp ) );
//    prop.major = 1;
//    prop.minor = 3;
//    HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );
//    printf( "ID of CUDA device closest to revision 1.3:  %d\n", dev );

//    HANDLE_ERROR( cudaSetDevice( dev ) );

    int c;
    int *dev_c;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );
    add<<<1,1>>>(2,7,dev_c);
    HANDLE_ERROR( cudaMemcpy( &c, dev_c, sizeof(int),
                              cudaMemcpyDeviceToHost ) );
    printf( "2 + 7 = %d\n", c );
    HANDLE_ERROR( cudaFree( dev_c ) );

    return 0;
}
