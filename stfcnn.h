#ifndef STFCNN_H
#define STFCNN_H

/* stfcnn - supervised-training, fully-connected neural network */

/* The struct stfcnn represents the neural network, and holds it's state at all
   times, as well as its inputs and outputs.

   In order for the network to map the values of each internal calculation to a
   value between 0 and 1, an activation function is needed, which needs to be
   provided to the network on creation.
   The derivative of this function is also needed as it's used in the
   calculations done to adjust the weights of each axon in the learning
   process. Default activation functions sigmoid and its derivative are
   provided in utils.h */

struct stfcnn
{
	int layers;						/* number of layers including input and
									   output layers */
	int *layer_size;				/* number of nodes in each layer, including
									   bias node */
	double **nodes;					/* value of each node on each layer,
									   including bias node */
	double ***axons;				/* value of each axon costfcnnecting a pair
									   of nodes */
	double **error;					/* calculated error of each node compared
									   to its desired value */
	int *target;					/* target values we want to achieve in
									   output layer */
	double learning_factor;			/* factor by which wheight corrections are
									   multiplied */
	double (*act_fcn)(double);		/* pointer to the activation function
									   used */
	double (*drv_act_fcn)(double);	/* derivative of the activation function */
	double *input;					/* copy of first layer nodes (except
									   bias), used for ease of access*/
	double *output;					/* copy of last layer nodes, used for
									   ease of access */
	double cost;					/* value of the cost function for the
									   results given by the network */
	int answer;						/* The most confident output index */
	double confidence;				/* The confidence of the answer */
};

/* This function allocates the memory needed for a network of the size
   specified. Arguments are the number of layers and an array with the number
   of nodes in each layer (excluding bias nodes) */
struct stfcnn create_stfcnn(int layers, int *l);
/* Initialization function, gives values to bias nodes and random weights to
   axons */
void init_values(struct stfcnn n);
/* Deallocation of memory of the network */
void free_stfcnn(struct stfcnn network);
/* Writing and loading state of network to a file to save a trained network for
   production */
int save_state(struct stfcnn nn, char *file);
struct stfcnn load_state(char *file);

/* Propagation forwards function. Analysis of the input and calculation of an
   output. Thinking function. */
void think(struct stfcnn *nn);
/* Backpropagation of the error and changes in the weights of each axon.
   Learning function */
void learn(struct stfcnn *nn);

#endif /* STFCNN_H */
