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
            Structure = file_maneger.GetStructure();
            input = new Layer(file_maneger, Structure, 1);
        }

        public double[] ComputeNetwork(double[] input) => buffer = this.input.PropegateDown(input);

        public double ComputeCost(double[] expected)
        {
            double ret = 0;
            for (int i = 0; i < buffer.Length; i++)
            {
                ret += Math.Pow(buffer[i] - expected[i], 2);
            }
            return ret;
        }

        public void Export(string location)
        {
            File_Maneger file_maneger = new File_Maneger(location);
            file_maneger.Erase();
            file_maneger.WriteStructure(Structure);
            input.Export(file_maneger);
        }
    }
}
