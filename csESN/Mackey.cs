using System;
using NumpyDotNet;

namespace csESN
{
    public class Mackey
    {
        public static void Run()
        {
            var data = np.fromfile("./data/mackey_glass_t17.npy", dtype: np.Float64); //  http://minds.jacobs-university.de/mantas/code
            data = (ndarray)data["10:"]; // hack or bug. mackey_glass_t17.npy was saved by numpy. 
                                         // NumpyDotNet np.fromfile load first 10 itens that are not loaded by numpy np.load.
            var esn = new ESN(n_inputs: 1,
                      n_outputs: 1,
                      n_reservoir: 500,
                      spectral_radius: 1.5,
                      random_state: 42);

            int trainlen = 2000;
            int future = 2000;
            var pred_training = esn.fit(np.ones(trainlen), (ndarray)data[$":{trainlen}"]);
            var prediction = esn.predict(np.ones(future));
            ndarray sqr = np.power(prediction.flatten() - data[$"{trainlen}:{trainlen + future}"], 2);
            System.Console.WriteLine("test error: \n" + np.sqrt((ndarray)np.mean(sqr)));
        }
    }
}