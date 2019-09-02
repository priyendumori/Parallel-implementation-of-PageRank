/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Title :- Google Page Rank using Power method Cuda
Authors :- Rushitkumar Jasani (2018201034)
		:- Priyendu Mori (2018201103)
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

#include <bits/stdc++.h>
using namespace std;

const int V = 1000;

#define ll int

float *output = NULL;
float graph[V][V];

fstream inp("input.txt", ios::in);
fstream out("output.txt", ios::out);

void build_matrix(float graph[V][V])
{
	ll source;
	ll dest;

	while (!inp.eof())
	{
		inp >> source >> dest;
		graph[source - 1][dest - 1] = 1;
	}

	for (ll i = 0; i < V; i++)
	{
		ll sum = 0;
		for (ll j = 0; j < V; j++)
		{
			sum += graph[i][j];
		}
		for (ll j = 0; j < V; j++)
		{
			if (sum != 0)
				graph[i][j] /= sum;
			else
				graph[i][j] = (1.0 / (float)V);
		}
	}
}

void transpose(float graph[V][V])
{
	float temp;
	for (ll i = 0; i < V; i++)
	{
		for (ll j = 0; j <= i; j++)
		{
			temp = graph[i][j];
			graph[i][j] = graph[j][i];
			graph[j][i] = temp;
		}
	}
}

float norm_2(float vect[V])
{
	float norm2 = 0.0;
	for (ll i = 0; i < V; i++)
		norm2 += vect[i] * vect[i];
	return sqrt(norm2);
}

float norm_1(float vect[V])
{
	float norm1 = 0.0;
	for (ll i = 0; i < V; i++)
		norm1 += vect[i];
	return norm1;
}

void power(float graph[V][V])
{
	float vect[V];
	output = (float *)malloc(sizeof(float) * V);
	float prev[V];
	for (ll i = 0; i < V; i++)
	{
		vect[i] = 0.85;
		prev[i] = vect[i];
	}

	float norm2 = norm_1(vect);

	for (ll i = 0; i < V; i++)
		vect[i] /= norm2;

	ll count = 1000;
	while (count--)
	{
		for (ll i = 0; i < V; i++)
		{
			float sum = 0.0;
			for (ll j = 0; j < V; j++)
			{
				sum += prev[j] * graph[i][j];
			}
			vect[i] = sum;
		}

		float norm1 = norm_1(vect);

		for (ll i = 0; i < V; i++)
		{
			vect[i] /= norm1;
			prev[i] -= vect[i];
		}

		if (norm_1(prev) < 0.000001)
		{
			norm1 = norm_1(vect);

			for (ll i = 0; i < V; i++)
				output[i] = vect[i] / norm1;
			break;
		}
		else
		{
			for (ll i = 0; i < V; i++)
				prev[i] = vect[i];
		}
	}
	float norm1 = norm_1(vect);

	for (ll i = 0; i < V; i++)
		output[i] = vect[i] / norm1;
	return;
}

void display_rank()
{
	pair<float, ll> P;
	priority_queue<pair<float, ll>> pq;

	for (ll i = 0; i < V; i++)
	{
		P = make_pair(output[i], i);
		pq.push(P);
	}

	ll i = 1;
	ll x = 10;
	while (!pq.empty() && x--)
	{
		P = pq.top();
		printf("\nRank[%d] :%d", i, P.second + 1);
		string s = "\nRank[" + to_string(i) + "] :" + to_string(P.second + 1);
		out << s;
		pq.pop();
		i++;
	}
}

int main()
{
	build_matrix(graph);
	transpose(graph);
	clock_t start, end;
	start = clock();
	power(graph);
	end = clock();
	display_rank();
	printf("Time taken :%f\n", float(end - start));
	return 0;
}