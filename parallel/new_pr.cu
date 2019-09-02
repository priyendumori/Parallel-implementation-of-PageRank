/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Title :- Google Page Rank using Power method Cuda
Authors :- Rushitkumar Jasani (2018201034)
		:- Priyendu Mori (2018201103)
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

#define numberOfVertex 1000

const int V = numberOfVertex;
const int X = 32;
float graph[V][V];				// to store graph  	
float *output1 = NULL;
float *d_Sum_Of_Degree = NULL;
float *d_PR = NULL;
float *d_prev = NULL;
float *d_out = NULL;
float *d_Graph = NULL;

fstream inp("input.txt",ios::in);
fstream out("Output_para.txt",ios::out); 

void build_matrix(float graph[V][V])
{
	int source;
	int dest;
	while(!inp.eof()) 				// taking input from file
	{
		inp>> source >> dest;
		graph[source - 1][dest - 1] = 1;
	}		
}

void transpose(float graph[V][V])
{
	float temp;
	for (int i = 0; i < V; i++)
	{
		for (int j = 0; j <= i; j++)
		{
			temp = graph[i][j];
			graph[i][j] = graph[j][i];
			graph[j][i] = temp;
		}
	}
}

float norm(float vect[V]) 				// find norm
{
	float norm1 = 0.0;
	for(int i=0;i<V;i++)
		norm1 += vect[i];
	return norm1;
}

__global__ void calculateSumOfOutDegree(float * sumOfOutDegree, float* Graph)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int s;
    if (i < numberOfVertex )
    {
		// printf("THIS IS INSIDE i : %d\n", i);
        // sumOfOutDegree[i]  = 0;
		
		s = 0;
        for (int j = 0; j < numberOfVertex ; ++j)
        {
            // sumOfOutDegree[i] += *(Graph +i*numberOfVertex  +j);
			// printf(" - %d - ", *(Graph + i*numberOfVertex + j));
			s += *(Graph + i*numberOfVertex + j);
			// sumOfOutDegree[i] += Graph[i][j];
        }
		// printf("this is s : %d\n",s);
		for(int j=0;j<numberOfVertex;j++)
		{
			if(s!= 0)
				*(Graph +i*numberOfVertex + j) /= s;
			else 							// if node has no outlink.
				*(Graph +i*numberOfVertex + j) = (1.0/(float)numberOfVertex);
		}
    }
}


__global__ void multi(float *vec, float *mat, float *out, const int V)  	// parallel matrix multiplication
{
    int tid=threadIdx.x+(blockIdx.x*blockDim.x);
    float sum=0;
    
    if(tid<numberOfVertex)
    {
        for(int i=0; i<numberOfVertex; i++)
            sum += vec[i]*mat[(tid*numberOfVertex)+i];
        out[tid]=sum;
    }
}

__global__ void compute_delta(float *vec, float *prev_vec, float norm)  	
{
    int tid=threadIdx.x+(blockIdx.x*blockDim.x);
    
    if(tid<numberOfVertex)
    {
        prev_vec[tid] -= (vec[tid]/norm);
    }

}

__global__ void assign_prev(float *vec, float *prev_vec)  	
{
    int tid=threadIdx.x+(blockIdx.x*blockDim.x);
    
    if(tid<numberOfVertex)
    {
        prev_vec[tid] = vec[tid];
    }
}


void power(float graph[V][V])
{
	float vect[V];
	float *output = (float*)malloc(sizeof(float)*V);
	float pre[V];
	for(int i=0;i<V;i++) 		// generating initial vector
	{	
		vect[i] = (1.0/(float)V);
		//vect[i] = 0.85;
		pre[i] = vect[i];
	}

	float norm1 = norm(vect);   // normalize the initial vector 
	for(int i=0;i<V;i++)
		vect[i]/=norm1; 

	cudaError_t err = cudaSuccess;
	err = cudaMemcpy(d_Graph, graph, numberOfVertex *V*sizeof(float), cudaMemcpyHostToDevice);   // transfer graph to GPU
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy graph from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(d_PR, vect, V*sizeof(float), cudaMemcpyHostToDevice); 					// transfer vector to GPU

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	int count = 1000;
	output1 = (float*)malloc(sizeof(float)*V);
	output = (float*)malloc(sizeof(float)*V);
	while(count--)	  							// if power method don't converge
	{	  	

    	multi<<<V/X + 1,X>>>(d_PR, d_Graph, d_out, V);
    	cudaMemcpy(output, d_out, sizeof(float)*V, cudaMemcpyDeviceToHost); //transfer multiplied vector to CPU 

		float norm1 = norm(output);  // normalize the output vector
        cudaMemcpy(d_prev, pre, sizeof(float)*V, cudaMemcpyHostToDevice);
        compute_delta<<< V/X + 1, X >>>(d_out, d_prev, norm1);
        cudaMemcpy(pre, d_prev, sizeof(float)*V, cudaMemcpyDeviceToHost);

		if(norm(pre) < 0.000000001) 
		{	
			norm1 = norm(output);

			for(int i=0;i<V;i++)
				output1[i] = output[i]/norm1;
			break;
		}
		else 
		{
            assign_prev<<< V/X + 1, X >>>(d_out, d_prev);
            cudaMemcpy(pre, d_prev, sizeof(float)*V, cudaMemcpyDeviceToHost);
		}	
	}	

	norm1 = norm(output);
	
	for(int i=0;i<V;i++)
		output1[i] = output[i]/norm1;
	return;
}

void display_rank()
{
	pair<float,int> P;
	priority_queue< pair<float,int> > pq;

	for(int i=0;i<V;i++)
	{
		P = make_pair(output1[i],i);
		pq.push(P);
	}	

	int i = 1;
	while(!pq.empty())
	{
		P = pq.top();
		printf("\nRank[%d] :%d", i, P.second + 1);
		out<<"\nRank[" << i << "] :" << P.second + 1; 
		pq.pop();
		i++;
	}		

}

int main(int argc,char* argv[])
{
	cudaError_t err = cudaSuccess;
    
    size_t size = numberOfVertex  * sizeof(float);  		//size of vectors or matrices

    cudaMalloc((void **)&d_Sum_Of_Degree, size);
    cudaMalloc((void**)&d_PR,size);
    cudaMalloc((void**)&d_out,size);
    cudaMalloc((void**)&d_prev,size);
    cudaMalloc((void **)&d_Graph, size * numberOfVertex );
   
    if(d_Sum_Of_Degree == NULL || d_PR == NULL || d_Graph == NULL)
    {
        cout << "Failed to allocate memory on the device"<<endl;
    }

	build_matrix(graph);			// populate graphs and process it
	cudaMemcpy(d_Graph, graph, size * numberOfVertex, cudaMemcpyHostToDevice);
    calculateSumOfOutDegree<<< V/X + 1,X >>>(d_Sum_Of_Degree, d_Graph);
    cudaDeviceSynchronize();
    cudaMemcpy(graph, d_Graph, size * numberOfVertex, cudaMemcpyDeviceToHost);
   	transpose(graph);				// transposing to make it in correct form

   	clock_t start,end;
   	start = clock();
    power(graph); 		// Parallel Power method
  	end = clock();
	display_rank();
	
	printf("Time taken :%f\n",float(end - start));

	cudaFree(d_Sum_Of_Degree);
    cudaFree(d_PR);
    cudaFree(d_out);
    cudaFree(d_Graph);
    
	return 0;
}
