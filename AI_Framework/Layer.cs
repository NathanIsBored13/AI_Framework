using System;

namespace AI_Framework
{
    class Layer
    {
        private readonly double[,] weights;
        private readonly double[] biases;
        private readonly int depth;
        private readonly Layer child;

        public Layer(int[] structure, int depth)
        {
            this.depth = depth;
            if (depth != structure.Length - 1) child = new Layer(structure, depth + 1);
            else child = null;
            biases = new double[structure[depth]];
            weights = new double[structure[depth], structure[depth - 1]];
            Random random = new Random();
            for (int x = 0; x < structure[depth]; x++)
            {
                biases[x] = random.NextDouble() - 0.5;
                for (int y = 0; y < structure[depth - 1]; y++)
                {
                    weights[x, y] = random.NextDouble() - 0.5;
                }
            }
        }

        public Layer(File_Maneger file_maneger, int[] structure, int depth)
        {
            this.depth = depth;
            weights = new double[structure[depth], structure[depth - 1]];
            biases = new double[structure[depth]];
            file_maneger.LoadNext(weights, biases);
            if (file_maneger.IsMore()) child = new Layer(file_maneger, structure, depth + 1);
        }

        public Layer(Neural_Network_Copy_Reference reference, int depth)
        {
            biases = (double[])Neural_Network.DeepClone(reference.biases[depth]);
            weights = (double[,])Neural_Network.DeepClone(reference.weights[depth]);
            this.depth = ++depth;
            if (depth < reference.biases.GetLength(0)) child = new Layer(reference, depth);
        }

        public void InputCopyData(Neural_Network_Copy_Reference reference, int depth)
        {
            reference.biases[depth] = biases;
            reference.weights[depth] = weights;
            if (child != null) child.InputCopyData(reference, depth + 1);
        }

        public double[] PropegateDown(double[] activation)
        {
            double[] buffer = new double[weights.GetLength(0)];
            for (int x = 0; x < weights.GetLength(0); x++)
            {
                for (int y = 0; y < weights.GetLength(1); y++)
                {
                    buffer[x] += activation[y] * weights[x, y];
                }
                buffer[x] = Math.Tanh(biases[x] + buffer[x]);
            }
            if (child != null) buffer = child.PropegateDown(buffer);
            return buffer;
        }

        public void Nudge(double chance, double max_delta)
        {
            max_delta *= 2;
            Random random = new Random();
            for (int x = 0; x < weights.GetLength(0); x++)
            {
                biases[x] = random.NextDouble() <= chance ? random.NextDouble() * max_delta - 1 : biases[x];
                for (int y = 0; y < weights.GetLength(1); y++)
                {
                    weights[x, y] = random.NextDouble() <= chance ? random.NextDouble() * max_delta - 1 : weights[x, y];
                }
            }
            if (child != null) child.Nudge(chance, max_delta);
        }

        public void Export(File_Maneger file_Manager)
        {
            file_Manager.Export(depth, weights, biases);
            if (child != null) child.Export(file_Manager);
        }
    }
}
