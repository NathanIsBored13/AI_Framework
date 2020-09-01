using System;
using System.Collections;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace AI_Framework
{
    public struct Neural_Network_Copy_Reference
    {
        public readonly double[][] biases;
        public readonly double[][,] weights;

        public Neural_Network_Copy_Reference(int length)
        {
            biases = new double[length][];
            weights = new double[length][,];
        }
    }

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

        public Neural_Network(Neural_Network_Copy_Reference reference, int[] structure)
        {
            Structure = (int[])DeepClone(structure);
            input = new Layer(reference, 0);
        }

        public Neural_Network Copy()
        {
            Neural_Network_Copy_Reference reference = new Neural_Network_Copy_Reference(Structure.Length - 1);
            input.InputCopyData(reference, 0);
            return new Neural_Network(reference, Structure);
        }

        public static object DeepClone(object obj)
        {
            using (var ms = new MemoryStream())
            {
                if (obj == null)
                {
                    Console.WriteLine("null");
                }
                var formatter = new BinaryFormatter();
                formatter.Serialize(ms, obj);
                ms.Position = 0;

                return formatter.Deserialize(ms);
            }
        }


        public double[] ComputeNetwork(double[] input) => buffer = this.input.PropegateDown(input);

        public double[] GetLastResult() => buffer;

        public void Nudge(double chance, double max_delta) => input.Nudge(chance, max_delta);

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
