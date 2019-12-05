using System;
using System.Collections.Generic;
using System.Text;

namespace AI_Framework
{
    class Layer
    {
        private Layer parent, child;
        private int node_width;
        private double[] biases;
        private double[,] weights;
        private Random random;
        public Layer(Layer parent, int[] structure, int index)
        {
            node_width = structure[index];
            this.parent = parent;
            if (index != structure.Length - 1) child = new Layer(this, structure, index + 1);
            else child = null;
            biases = new double[structure[index]];
            weights = new double[structure[index], structure[index - 1]];
            random = new Random();
            for (int x = 0; x < structure[index]; x++)
            {
                biases[x] = random.NextDouble() * 10d - 5d;
                for (int y = 0; y < structure[index - 1]; y++)
                {
                    weights[x, y] = random.NextDouble() * 10d - 5d;
                }
            }
        }
        public double[] Propegate_Down(double[] activation)
        {
            double[] buffer = new double[node_width];
            for (int x = 0; x < node_width; x++)
            {
                for (int y = 0; y < activation.Length; y++)
                {
                    buffer[x] += activation[y] * weights[x, y];
                }
                buffer[x] = 1d / (1d + Math.Exp(biases[x] - buffer[x]));
            }
            if (child != null) buffer = child.Propegate_Down(buffer);
            return buffer;
        }
    }
}
