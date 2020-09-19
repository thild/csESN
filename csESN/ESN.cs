using System;
using System.Linq;
using System.Numerics;
using MathNet.Numerics.LinearAlgebra;
using NumpyDotNet;
using NumpyDotNet.RandomAPI;

namespace csESN
{


    public class ESN
    {

        //  """
        // Args:
        //     n_inputs: nr of input dimensions
        //         n_outputs: nr of output dimensions
        //         n_reservoir: nr of reservoir neurons
        //         spectral_radius: spectral radius of the recurrent weight matrix
        //     sparsity: proportion of recurrent weights set to zero
        //     noise: noise added to each neuron(regularization)
        //         input_shift: scalar or vector of length n_inputs to add to each
        //                     input dimension before feeding it to the network.
        //     input_scaling: scalar or vector of length n_inputs to multiply
        //                 with each input dimension before feeding it to the netw.
        //     teacher_forcing: if True, feed the target back into output units
        //     teacher_scaling: factor applied to the target signal
        //     teacher_shift: additive term applied to the target signal
        //     out_activation: output activation function (applied to the readout)
        //         inverse_out_activation: inverse of the output activation function
        //         random_state: positive integer seed, np.rand.RandomState object,
        //                   or None to use numpy's builting RandomState.
        //     silent: supress messages
        // """
        public ESN(int n_inputs, int n_outputs, int n_reservoir = 200,
                     double spectral_radius = 0.95, double sparsity = 0, double noise = 0.001,
                     ndarray input_shift = null,
                     ndarray input_scaling = null,
                     bool teacher_forcing = true,
                     ndarray feedback_scaling = null,
                     double teacher_scaling = 0, double teacher_shift = 0,
                     Func<ndarray, ndarray> out_activation = null,
                     Func<ndarray, ndarray> inverse_out_activation = null,
                     object random_state = null,
                     bool silent = true)
        {

            if (out_activation == null) out_activation = IdentityFunction<ndarray>();
            if (inverse_out_activation == null) inverse_out_activation = IdentityFunction<ndarray>();
            // check for proper dimensionality of all arguments and write them down.
            this.n_inputs = n_inputs;
            this.n_reservoir = n_reservoir;
            this.n_outputs = n_outputs;
            this.spectral_radius = spectral_radius;
            this.sparsity = sparsity;
            this.noise = noise;
            this.input_shift = correct_dimensions(input_shift, n_inputs);
            this.input_scaling = correct_dimensions(input_scaling, n_inputs);

            this.teacher_scaling = teacher_scaling;
            this.teacher_shift = teacher_shift;

            this.out_activation = out_activation;
            this.inverse_out_activation = inverse_out_activation;

            // the given random_state might be either an actual RandomState object,
            // a seed or None (in which case we use numpy's builtin RandomState)
            if (random_state is RandomState) {
                this.random_state_ = random_state as np.random;
            }
            else if (random_state is int) {
                try
                {
                    var rnd = new np.random();
                    rnd.seed((int)random_state);
                    this.random_state_ = rnd;
                }
                catch (System.Exception ex)
                {
                    throw new Exception("Invalid seed: " + ex);
                }
            }
            else {
                this.random_state_ = new np.random();
            }
                
            this.teacher_forcing = teacher_forcing;
            this.silent = silent;
            this.initweights();
        }

        public int n_inputs { get; }
        public int n_reservoir { get; }
        public int n_outputs { get; }
        public double spectral_radius { get; }
        public double sparsity { get; }
        public double noise { get; }
        public ndarray input_shift { get; }
        public ndarray input_scaling { get; }
        public double teacher_scaling { get; }
        public double teacher_shift { get; }
        public Func<ndarray, ndarray> out_activation { get; }
        public Func<ndarray, ndarray> inverse_out_activation { get; }
        public RandomState random_state { get; }
        public np.random random_state_ { get; }
        public bool teacher_forcing { get; }
        public bool silent { get; }
        public ndarray W { get; private set; }
        public ndarray W_in { get; private set; }
        public ndarray W_feedb { get; private set; }
        public ndarray inputs_scaled { get; private set; }
        public ndarray teachers_scaled { get; private set; }
        public ndarray W_out { get; private set; }
        public ndarray laststate { get; private set; }
        public ndarray lastinput { get; private set; }
        public ndarray lastoutput { get; private set; }

        private void initweights()
        {
            // initialize recurrent weights:
            // begin with a random matrix centered around zero:
            var W = random_state_.rand(new shape(this.n_reservoir, this.n_reservoir)) - 0.5;
            // delete the fraction of connections given by (this.sparsity):
            W[random_state_.rand(W.shape) < this.sparsity] = 0;
            // compute the spectral radius of these weights:
            
            var array = W.AsComplexArray();
            var matrix = Matrix<Complex>.Build.DenseOfColumnMajor((int)W.shape.iDims[0], (int)W.shape.iDims[1], array);
            var evd = matrix.Evd().EigenValues.OrderBy(m => m.Real).ToArray();
            var ev = np.array(evd);

            var radius = (Complex)np.max(np.absolute(ev));

            // rescale them to reach the requested spectral radius:
            this.W = W * (this.spectral_radius / radius);

            // random input weights:
            this.W_in = random_state_.rand(new shape(this.n_reservoir, this.n_inputs)) * 2 - 1;
            // random feedback (teacher forcing) weights:
            this.W_feedb = random_state_.rand(new shape(this.n_reservoir, this.n_outputs)) * 2 - 1;
        }

        private ndarray _update(ndarray state, ndarray input_pattern, ndarray output_pattern)
        {
            // """performs one update step.

            // i.e., computes the next network state by applying the recurrent weights
            // to the last state & and feeding in the current input and output patterns
            // """
            ndarray preactivation;
            if (this.teacher_forcing)
            {
                var a = np.dot(this.W, state);
                var b = np.dot(this.W_in, input_pattern);
                var c = np.dot(this.W_feedb, output_pattern);
                preactivation = a + b + c;
            }
            else
            {
                preactivation = (np.dot(this.W, state)
                                 + np.dot(this.W_in, input_pattern));
            }
            return (np.tanh(preactivation)
                    + this.noise * (random_state_.rand(new shape(this.n_reservoir)) - 0.5));
        }

        private ndarray _scale_inputs(ndarray inputs)
        {
            // """for each input dimension j: multiplies by the j'th entry in the
            // input_scaling argument, then adds the j'th entry of the input_shift
            // argument."""
            if (this.input_scaling != null)
                inputs = np.dot(inputs, np.diag(this.input_scaling));
            if (this.input_shift != null)
                inputs = inputs + this.input_shift;
            return inputs;
        }
        private ndarray _scale_teacher(ndarray teacher)
        {
            //     """multiplies the teacher/target signal by the teacher_scaling argument,
            // then adds the teacher_shift argument to it."""
            if (this.teacher_scaling != 0)
                teacher = teacher * this.teacher_scaling;
            if (this.teacher_shift != 0)
                teacher = teacher + this.teacher_shift;
            return teacher;
        }
        private ndarray _unscale_teacher(ndarray teacher_scaled)
        {
            // """inverse operation of the _scale_teacher method."""
            if (this.teacher_shift != 0)
                teacher_scaled = teacher_scaled - this.teacher_shift;
            if (this.teacher_scaling != 0)
                teacher_scaled = teacher_scaled / this.teacher_scaling;
            return teacher_scaled;
        }

        public ndarray fit(ndarray inputs, ndarray outputs, bool inspect = false)
        {
            // """
            // Collect the network's reaction to training data, train readout weights.

            // Args:
            //     inputs: array of dimensions(N_training_samples x n_inputs)
            //         outputs: array of dimension(N_training_samples x n_outputs)
            //         inspect: show a visualisation of the collected reservoir states

            // Returns:
            //     the network's output on the training data, using the trained weights
            // """
            // transform any vectors of shape (x,) into vectors of shape (x,1):
            if (inputs.ndim < 2)
                inputs = np.reshape(inputs, new shape(inputs.shape.iDims[0], -1));
            if (outputs.ndim < 2)
                outputs = np.reshape(outputs, new shape(outputs.shape.iDims[0], -1));
            // transform input and teacher signal:
            inputs_scaled = this._scale_inputs(inputs);
            teachers_scaled = this._scale_teacher(outputs);

            if (!this.silent)
                System.Console.WriteLine("harvesting states...");
            // step the reservoir through the given input,output pairs:
            var states = np.zeros(new shape(inputs.shape.iDims[0], this.n_reservoir));
            foreach (var n in Enumerable.Range(1, (int)inputs.shape.iDims[0] - 1))
            {
                states[n, ":"] = this._update((ndarray)states[n - 1],
                                    (ndarray)inputs_scaled[n, ":"],
                                    (ndarray)teachers_scaled[n - 1, ":"]);
            }


            // learn the weights, i.e. find the linear combination of collected
            // network states that is closest to the target output
            if (!this.silent)
                System.Console.WriteLine("fitting...");
            // we'll disregard the first few states:
            var transient = Math.Min((int)(inputs.shape.iDims[1] / 10), 100);
            // include the raw inputs:
            var extended_states = np.hstack(new object[] { states, inputs_scaled });
            // Solve for W_out:
            ndarray es = (ndarray)extended_states[$"{transient}:", ":"];
            var array = ((ndarray)es).AsDoubleArray();
            var matrix = Matrix<double>.Build.DenseOfRowMajor((int)es.shape.iDims[0], (int)es.shape.iDims[1], array);
            matrix = matrix.PseudoInverse();
            es = np.array(matrix.ToArray());
            // es = es.reshape((int)es.shape.iDims[1], (int)es.shape.iDims[0]);
            var b = this.inverse_out_activation((ndarray)teachers_scaled[$"{transient}:", ":"]);
            this.W_out = np.dot(es, b).T;

            // remember the last state for later:
            this.laststate = (ndarray)states[-1, ":"];
            this.lastinput = (ndarray)inputs[-1, ":"];
            this.lastoutput = (ndarray)teachers_scaled[-1, ":"];


            // // optionally visualize the collected states
            // if (inspect)
            // {
            //     // (^-- we depend on matplotlib only if this option is used)
            //     plt.figure(
            //         figsize = (states.shape[0] * 0.0025, states.shape[1] * 0.01));
            //         plt.imshow(extended_states.T, aspect = 'auto',
            //             interpolation = 'nearest');
            //     plt.colorbar();
            // }

            if (!this.silent)
                System.Console.WriteLine("training error:");
            // apply learned weights to the collected states:
            var pred_train = this._unscale_teacher(this.out_activation(
                np.dot(extended_states, this.W_out.T)));
            if (!this.silent)
            {
                var diff = (pred_train - outputs) ^ 2;
                np.sqrt((ndarray)np.mean(diff));
                System.Console.WriteLine();
            }
            return pred_train;
        }

        public ndarray predict(ndarray inputs, bool continuation = true)
        {
            //     """
            // Apply the learned weights to the network's reactions to new input.

            // Args:
            //     inputs: array of dimensions(N_test_samples x n_inputs)
            //         continuation: if True, start the network from the last training state

            // Returns:
            //     Array of output activations
            // """
            if (inputs.ndim < 2)
                inputs = np.reshape(inputs, new shape(inputs.shape.iDims[0], -1));
            var n_samples = inputs.shape.iDims[0];

            if (continuation)
            {
                laststate = this.laststate;
                lastinput = this.lastinput;
                lastoutput = this.lastoutput;
            }
            else
            {
                laststate = np.zeros(this.n_reservoir);
                lastinput = np.zeros(this.n_inputs);
                lastoutput = np.zeros(this.n_outputs);
            }

            inputs = np.vstack(new object[] { lastinput, this._scale_inputs(inputs) });
            var states = np.vstack(new object[] { laststate, np.zeros(new shape(n_samples, this.n_reservoir)) });
            var outputs = np.vstack(new object[] { lastoutput, np.zeros(new shape(n_samples, this.n_outputs)) });

            foreach (var n in Enumerable.Range(0, (int)n_samples))
            {
                states[n + 1, ":"] = this._update((ndarray)states[n, ":"],
                    (ndarray)inputs[n + 1, ":"], (ndarray)outputs[n, ":"]);
                outputs[n + 1, ":"] =
                    this.out_activation(np.dot(this.W_out, np.concatenate((states[n + 1, ":"], inputs[n + 1, ":"]))));
            }

            return this._unscale_teacher(this.out_activation((ndarray)outputs["1:"]));
        }
        private ndarray correct_dimensions(ndarray s, int targetlength)
        {
            // """checks the dimensionality of some numeric argument s, broadcasts it
            //    to the specified length if possible.

            // Args:
            //     s: None, scalar or 1D array
            //     targetlength: expected length of s

            // Returns:
            //     None if s is None, else numpy vector of length targetlength
            // """

            if (s != null)
            {
                s = np.array(s);
                if (s.ndim == 0)
                {
                    s = np.array(s * targetlength);
                }
                else if (s.ndim == 1)
                {
                    if (s.dims[0] != targetlength)
                    {
                        throw new Exception("arg must have length " + targetlength);
                    }
                }
                else
                    throw new Exception("Invalid argument");
            }
            return s;
        }

        public static Func<T, T> IdentityFunction<T>()
        {
            return x => x;
        }

    }
}