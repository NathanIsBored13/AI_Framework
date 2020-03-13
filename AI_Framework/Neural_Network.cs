using System;

namespace AI_Framework
{
    class Neural_Network
    {
        public int[] Structure { get; }
        private readonly Layer input;
        private double[] buffer;

        public Neural_Network(int[] structure)
        {
            Structure = structure;
            input = new Layer(structure, 1);
            buffer = new double[structure[structure.Length - 1]];
        }

        public Neural_Network(string location)
        {
            File_Maneger file_maneger = new File_Maneger(location);
            input = new Layer(file_maneger, 1);
            Structure = file_maneger.GetStructure();
        }

        public double[] ComputeNetwork(double[] input)
        {
            buffer = this.input.PropegateDown(input);
            return buffer;
        }

        public double ComputeCost(int correct, int min, int max)
        {
            double ret = 0;
            for (int i = 0; i < buffer.Length; i++) ret += Math.Pow(buffer[i] - (i == correct ? min : max), 2);
            return ret;

        }
        public void Export(string location)
        {
            File_Maneger file_maneger = new File_Maneger(location);
            file_maneger.Erase();
            Console.WriteLine($"Erased folder at {location}");
            input.Export(file_maneger);
            Console.WriteLine("Compleated export");
        }
    }
}
