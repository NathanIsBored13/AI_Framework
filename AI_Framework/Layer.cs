using System;

namespace AI_Framework
{
    class Layer
    {
        private double[,] weights;
        private double[] biases;
        private int depth { get; }
        private Layer child;
        public Layer(int[] structure, int depth)
        {
            this.depth = depth;
            Console.WriteLine("layer {0} created", depth);
            if (depth != structure.Length - 1) child = new Layer(structure, depth + 1);
            else child = null;
            biases = new double[structure[depth]];
            weights = new double[structure[depth], structure[depth - 1]];
            Random random = new Random();
            for (int x = 0; x < structure[depth]; x++)
            {
                biases[x] = random.NextDouble() * 10d - 5d;
                for (int y = 0; y < structure[depth - 1]; y++)
                {
                    weights[x, y] = random.NextDouble() * 10d - 5d;
                }
            }
        }
        public Layer(File_Maneger file_maneger, int depth)
        {
            this.depth = depth;
            LayerInfo buffer = file_maneger.ReadNext();
            weights = buffer.weights;
            biases = buffer.biases;
            Console.WriteLine("layer {0} loaded", depth);
            if (file_maneger.IsMore()) child = new Layer(file_maneger, depth + 1);
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
                buffer[x] = 1d / (1d + Math.Exp(biases[x] - buffer[x]));
            }
            if (child != null) buffer = child.PropegateDown(buffer);
            return buffer;
        }
        public void Export(File_Maneger file_Manager)
        {
            file_Manager.Export(depth, weights, biases);
            Console.WriteLine("Exported layer {0}", depth);
            if (child != null) child.Export(file_Manager);
        }
    }
}
