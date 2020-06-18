#include "stfcnn.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TRIES 1000

int main(int argc, char **argv)
{
	int ns[3] = {2,3,1};
	struct stfcnn net = create_stfcnn(3, ns, 0);
	net.learning_factor=0.1;
	net.activation_function=&sigmoid;
	net.deriv_activation_function=&sigmoid_deriv;
	/* Test the neural network by sorting values above/under x=y */

	/* Input values are x coord for I[0] and y coord for I[1] */
	/* Output value is 1 for x>y and 0 for x<y */

	/* Train TRIES points */
	srand(time(NULL));
	int i,j;
	double x,y;
	int ox, oy; /* original values */
	int correct = 0;
	int last_guess[100] = {0};
	int last_corrects;
	for(i=0;i<TRIES;i++)
	{
		ox = rand()%2001 - 1000;
		oy = rand()%2001 - 1000;
		x = (double) ox;
		y = (double) oy;
		/* escalate the input  between +-1 */
		x /= 1000.0;
		y /= 1000.0;
		net.input[0]=x;
		net.input[1]=y;
		if(x>y) net.target[0] = 1;
		else net.target[0] = 0;
		/* rotate last 100 guesses */
		for(j=0;j<99;j++) last_guess[j] = last_guess[j+1];
		/* add last guess */
		last_guess[99] = think(net,1);
		last_corrects = 0;
		for(j=0;j<100;j++) last_corrects += last_guess[j];
		/* if we are guessing right, reduce learning factor */
		if(last_corrects > 90)
		{
			net.learning_factor /= 1.5;
			for(j=0;j<100;j++) last_guess[j] = 0;
		}
	}
	for(i=0;i<TRIES;i++)
	{
		ox = rand()%2001 - 1000;
		oy = rand()%2001 - 1000;
		x = (double) ox;
		y = (double) oy;
		/* escalate the input  between +-1 */
		x /= 1000.0;
		y /= 1000.0;
		net.input[0]=x;
		net.input[1]=y;
		if(x>y) net.target[0] = 1;
		else net.target[0] = 0;
		think(net,0);
		if(net.output[0] < 0.5 && net.target[0] == 0) correct++;
		if(net.output[0] >= 0.5 && net.target[0] == 1) correct++;
	}
	printf("\nOut of %d tries, the network guessed %d times correctly (%0.2lf%%)\n", TRIES, correct, (double)correct/TRIES * 100);

	free_stfcnn(net);
	return 0;
}
