#include <stdlib.h> /* malloc(), free(), rand() */
#include <time.h> /* time() */
#include <stdio.h> /* FILE, fread(), fwrite() */

#include "stfcnn.h"

/* In this implementation we treat the bias term for the calculations
   as a node with a value equal to the bias (in this case, 1) and add
   it to the end of each layer. This node is special in that no node
   from the previous layer connects to it, but it does connect to each
   node in the next layer.
   This makes the code a bit ugly since the loops have to watch out
   for the last node in each layer except the last one and is
   (hopefully not) a source of mistakes and bugs, but it's preferable
   than to overcomplicate the calculations having to add the bias
   manually later, and since the use cases of this net usually run
   incomplete data sets to save time and processing power this
   approach seems to work fine. */
struct stfcnn create_stfcnn(int layers, int *l)
{
	struct stfcnn network;
	int i,j;

	network.layers = layers;
	network.layer_size = malloc(sizeof(int) * network.layers);
	for(i=0;i<network.layers;i++) network.layer_size[i] = l[i];
   	for(i=0;i<network.layers-1;i++) network.layer_size[i]++; /* add bias node */

	network.nodes = malloc(sizeof(double *) * network.layers);
	network.error = malloc(sizeof(double *) * network.layers);
	network.axons = malloc(sizeof(void *) * network.layers);

	for(i=0;i<network.layers;i++)
	{
		network.nodes[i] = malloc(sizeof(double) * network.layer_size[i]);
		network.error[i] = malloc(sizeof(double) * network.layer_size[i]);
	}

	for(i=0;i<network.layers - 1;i++)
	{
		network.axons[i] = malloc(sizeof(double *) * network.layer_size[i]);
		for(j=0;j<network.layer_size[i];j++)
		{
			/* axons from one layer dont connect to bias of next layer,
			   but last layer doesnt have a bias */
			network.axons[i][j] = malloc(sizeof(double) *
				(network.layer_size[i+1] - (i == network.layers - 2 ? 0 : 1)));
		}
	}

	network.target = malloc(sizeof(int) * network.layer_size[network.layers-1]);

	/* this values need to be provided by the user before using the net */
	network.learning_factor = 0;
	network.act_fcn = NULL;
	network.drv_act_fcn = NULL;

	network.input = malloc(sizeof(double) * network.layer_size[0] - 1);
	network.output = malloc(sizeof(double) * network.layer_size[network.layers-1]);

	init_values(network);
	return network;
}
void init_values(struct stfcnn n)
{
	int i,j,k;
	srand(time(NULL));

	for(i=0;i<n.layers-1;i++)
	{
		for(j=0;j<n.layer_size[i];j++)
		{
			n.nodes[i][j] = 0.0;
			for(k=0;k<n.layer_size[i+1]-(i==n.layers-2 ? 0 : 1);k++) /* Last layer doesn't have bias node */
				n.axons[i][j][k] = (double)(rand() % 201 - 100) / 100.0; /* -1 - 1 */
		}
		n.nodes[i][n.layer_size[i]-1] = 1.0; /* bias */
	}
	for(i=0;i<n.layer_size[n.layers-1];i++)
		n.nodes[n.layers-1][i] = 0.0;
}

void free_stfcnn(struct stfcnn network)
{
	int i,j;
	for(i=0;i<network.layers;i++)
	{
		free(network.nodes[i]);
		free(network.error[i]);
	}
	for(i=0;i<network.layers - 1;i++)
	{
		for(j=0;j<network.layer_size[i];j++)
			free(network.axons[i][j]);
		free(network.axons[i]);
	}
	free(network.nodes);
	free(network.error);
	free(network.axons);
	free(network.layer_size);
	free(network.target);
	free(network.input);
	free(network.output);
}

int save_state(struct stfcnn nn, char *file)
{
	/* This function saves the state of nn in a specified file
	   for later loading. */

	int i,j;
	FILE *fp = fopen(file, "w");
	if(fp == NULL) return -1;

	/* We write data in an order in which we can later read
	   and know the size of each element before reading it */

	/* layers is an int, and is fundamentally the most important */
	fwrite(&(nn.layers),sizeof(int),1,fp);
	/* once we know how many layers, write how many nodes in each layer */
	fwrite(nn.layer_size,sizeof(int),nn.layers,fp);
	/* now we know how many layers there are and how many nodes there
	   are in each layer, so write all the axons now */
	for(i=0;i<nn.layers-1;i++)
		for(j=0;j<nn.layer_size[i];j++)
			fwrite(nn.axons[i][j],sizeof(double),nn.layer_size[i+1],fp);

	/* the other parameters of the struct are not part of the "state" of
	   the network, so they can be discarded.
	   learning_factor, error, target, activation functions, etc. are
	   implementation specific and are not relevant to the state
	   of the network, they are just variables for calculating and
	   training which have to be implemented specifically by the user */

	/* Even the nodes are not considered part of the state, because
	   their value depends on the specific input provided, and the thing
	   that gets trained are the axon weights */

	fclose(fp);
	return 0;
}
struct stfcnn load_state(char *file)
{
	/* This function reads the state of nn from a specified file.
	   The struct nn will be (re)initialized completely, so data
	   may be lost and memory leaked if an already-initialized
	   network is used.
	   The created net only contains saved data, so in order for
	   the net to function properly, the rest of the data (such
	   as the activation functions) must be defined elsewhere
	   before using the net */

	int i,j;
	int lay, *lay_size;
	FILE *fp = fopen(file, "r");
	if(fp == NULL)
	{
		int l[2] = {2,2};
		struct stfcnn n = create_stfcnn(2,l);
		return n;
	}

	/* We read data in the order in which we wrote it, in a way
	   that allows us to know the size of each element before reading it */

	/* layers is an int, and is fundamentally the most important */
	fread(&lay,sizeof(int),1,fp);
	/* once we know how many layers, read how many nodes in each layer */
	lay_size = malloc(sizeof(int) * lay);
	fread(lay_size,sizeof(int),lay,fp);
	/* now we know how many layers there are and how many nodes there
	   are in each layer, so we can initialize the net now */
	struct stfcnn nn = create_stfcnn(lay, lay_size);
	/* with the net created we just need to load the axon values */
	for(i=0;i<nn.layers-1;i++)
		for(j=0;j<nn.layer_size[i];j++)
			fread(nn.axons[i][j],sizeof(double),nn.layer_size[i+1],fp);

	/* the other parameters of the struct are not part of the "state" of
	   the network, so they can be discarded.
	   learning_factor, error, target, activation functions, etc. are
	   implementation specific and are not relevant to the state
	   of the network, they are just variables for calculating and
	   training which have to be implemented specifically by the user */

	/* Even the nodes are not considered part of the state, because
	   their value depends on the specific input provided, and the thing
	   that gets trained are the axon weights */

	free(lay_size);
	fclose(fp);
	return nn;
}

void think(struct stfcnn *nn)
{
	int i,j,k;
	double sum;

	/* Before we start, copy input into first layer */
	for(i=0;i<nn->layer_size[0]-1;i++) nn->nodes[0][i] = nn->input[i]; /* dont overwrite bias */

	for(i=1;i<nn->layers;i++)
	{
		for(j=0;j<nn->layer_size[i]-(i==nn->layers-1 ? 0 : 1);j++) /* cant change value of bias, but last layer doesnt have bias*/
		{
			sum = 0;
			for(k=0;k<nn->layer_size[i-1];k++)
				sum += nn->nodes[i-1][k] * nn->axons[i-1][k][j];
			nn->nodes[i][j] = nn->act_fcn(sum);
		}
	}

	/* After we finish, for convenience copy last layer to output */
	for(i=0;i<nn->layer_size[nn->layers-1];i++) nn->output[i] = nn->nodes[nn->layers-1][i];

	/* Also calculate the cost function */
	nn->cost = 0;
	for(i=0;i<nn->layer_size[nn->layers-1];i++)
		nn->cost += (nn->output[i] - nn->target[i]) * (nn->output[i] - nn->target[i]);

	/* Return highest output as the selected one */
	int maxindex = -1;
	double max = -1;
	for(i=0;i<nn->layer_size[nn->layers-1];i++)
	{
		if(nn->output[i] > max)
		{
			max = nn->output[i];
			maxindex = i;
		}
	}
	nn->confidence = max;
	nn->answer = maxindex;
}

void learn(struct stfcnn *nn)
{
	/* NOTE: this is the function in which my knowledge is most limited, and
	   I have already have made mistakes before. If some of the deductions
	   made in the comments below (or in the sources mentioned) are wrong or
	   badly implemented, it means I really don't fully understand what I'm
	   talking about. So, modifications to this should be made with a lot of
	   planning beforehand */

	/* This function propagates output error backwards and
	   tunes each axon's weight to "learn" from the propagated
	   error */

	/* Doing partial derivatives to find out what is the effect
	   of each of the weighted axons on the value of the final
	   output nodes, and implementing gradient descent in order to arrive to an
	   optimal solution faster, we calculate the change in weight of each axon
	   as a fraction of the value of the partial derivative of the cost
	   function with respect to the weights.
	   For reference check video of 3blue1brown, neural networks part 4 about
	   the calculus of NN.

	   For the axons in the second to last layer, this expression goes as
	   follows:
	   delta = 2 * (output - target) *
	   		   der_act(output) *
			   value_of_node_this_axon_connects_from *
			   learning_factor

	   Which can also be written as:
	   delta = E *
	   		   der_act(node_right) *
			   node_left *
			   learning_factor

	   For the output layer, E is calculated directly from the target output,
	   2*(output-target). For the rest of the layers, the error E of a given
	   node is the sum of its contributions to the error of the nodes on the
	   next layer. This can be calculated as:
	   E = sum(E_next * der_act(next) * weight_axon_connecting_both_nodes)

	   We store the values of each error instead of recursively calculating
	   forwards because there is a number of nested loops (summaroties) equal
	   to the number of layers after the axon, so this way is simpler for
	   networks with more than 3 layers. Otherwise we can hardcode this for 3
	   layers like in carykh's racial neural network (github).

	   drv_act_fcn must correspond to  the derivative of act_fcn function used
	   to calculate the value of the nodes of nn */

	int curr_layer;
	int node;
	int i,j,k;

	/* Variable learning factor test */
	/* Maximum of cost function is the number of outputs */
/*	double cost_output = nn->cost / nn->layer_size[nn->layers-1];
	if(cost_output > 0.2)
		nn->learning_factor = 1;
	else if(cost_output > 0.1)
		nn->learning_factor = nn->cost * 4;
	else if(cost_output > 0.01)
		nn->learning_factor = nn->cost * 2;
	else
		nn->learning_factor = nn->cost;


	/* First we calculate the error for each node, while the value of each axon
	   is preserved */
	for(curr_layer=nn->layers-1;curr_layer>0;curr_layer--)
	{
		for(node=0;node<nn->layer_size[curr_layer]-(curr_layer==nn->layers-1 ? 0 : 1);node++)
		{
			if(curr_layer == nn->layers-1)
				nn->error[curr_layer][node] = 2 * (nn->nodes[curr_layer][node] - nn->target[node]);
			else
			{
				nn->error[curr_layer][node] = 0;
				for(i=0;i<nn->layer_size[curr_layer+1]-((curr_layer+1)==nn->layers-1 ? 0 : 1);i++)
					nn->error[curr_layer][node] += nn->error[curr_layer+1][i] * nn->drv_act_fcn(nn->nodes[curr_layer+1][i]) * nn->axons[curr_layer][node][i];
			}
		}
	}

	/* Then with each E calculated, we calculate the new value for each axon.
	   It really doesnt matter if we do this backwards or forwards, because the
	   errors are already calculated */

	for(curr_layer=nn->layers-2;curr_layer>=0;curr_layer--)
	{
		for(i=0;i<nn->layer_size[curr_layer];i++)
		{
			for(j=0;j<nn->layer_size[curr_layer+1] - ((curr_layer+1) == nn->layers-1 ? 0 : i); j++)
			{
				nn->axons[curr_layer][i][j] -=
					nn->error[curr_layer+1][j] *
					nn->drv_act_fcn(nn->nodes[curr_layer+1][j]) *
					nn->nodes[curr_layer][i] *
					nn->learning_factor;
			}
		}
	}

	/* Hardcoded for 3 layers */
/*
	double delta;
	for(i=0;i<nn->layer_size[0];i++)
	{
		for(j=0;j<nn->layer_size[1] - 1;j++)
		{
			delta = 0;
			for (k=0;k<nn->layer_size[2] - 1;k++)
			{
				delta +=
					2 * (nn->nodes[2][k] - nn->target[k]) *
					nn->nodes[2][k] * (1 - nn->nodes[2][k]) *
					nn->axons[1][j][k] *
					nn->nodes[1][j] * (1 - nn->nodes[1][j]) *
					nn->nodes[0][i] *
					nn->learning_factor;
			}
			nn->axons[0][i][j] -= delta;
	    }
	}
	for(i=0;i<nn->layer_size[1];i++)
	{
		for(j=0;j<nn->layer_size[2]-1;j++)
		{
			delta =
				2 * (nn->nodes[2][j] - nn->target[j]) *
				nn->nodes[2][j] * (1 - nn->nodes[2][j]) *
				nn->nodes[1][i] *
				nn->learning_factor;
            nn->axons[1][i][j] -= delta;
		}
	}
	*/
}
