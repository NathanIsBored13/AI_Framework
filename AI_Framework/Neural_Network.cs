using System;
using System.Collections.Generic;
using System.Text;

namespace AI_Framework
{
    struct LayerInfo
    {
        public double[,] weights;
        public double[] biases;
    }
    class Neural_Network
    {
        public int[] structure { get; }
        private double[] buffer;
        private Layer input;
        public Neural_Network(int[] structure)
        {
            this.structure = structure;
            input = new Layer(structure, 1);
            buffer = new double[structure[structure.Length - 1]];
        }
        public Neural_Network(string location)
        {
            File_Maneger file_maneger = new File_Maneger(location);
            input = new Layer(file_maneger, 1);
            structure = file_maneger.GetStructure();
        }
        public double[] ComputeNetwork(double[] input)
        {
            buffer = this.input.PropegateDown(input);
            return buffer;
        }
        public double ComputeCost(int correct)
        {
            double ret = 0;
            for (int i = 0; i < buffer.Length; i++) ret += Math.Pow(buffer[i] - (i == correct ? 1 : 0), 2);
            return ret;
        }
        public void Export(string location)
        {
            File_Maneger file_maneger = new File_Maneger(location);
            file_maneger.Erase();
            Console.WriteLine("Erased folder at {0}", location);
            input.Export(file_maneger);
            Console.WriteLine("Compleated export");
        }
    }
}
