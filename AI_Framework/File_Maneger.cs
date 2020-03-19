using System;
using System.Text;
using System.IO;

namespace AI_Framework
{
    class File_Maneger
    {
        private string Address { get; }
        private int pointer = 0;
        private readonly string[] files;

        public File_Maneger(string address)
        {
            Address = address;
            files = Directory.GetFiles(address, @"LAYER-*.ai");
        }

        public int[] GetStructure()
        {
            BinaryReader sr = new BinaryReader(File.Open($@"{Address}\Structure.ai", FileMode.Open));
            int[] ret = new int[sr.ReadInt32()];
            for (int i = 0; i < ret.Length; i++)
            {
                ret[i] = sr.ReadInt32();
            }
            sr.Close();
            return ret;
        }

        public void LoadNext(double[,] weights, double[] biases)
        {
            BinaryReader sr = new BinaryReader(File.Open(files[pointer++], FileMode.Open));
            for (int x = 0; x < weights.GetLength(0); x++)
            {
                for (int y = 0; y < weights.GetLength(1); y++)
                {
                    weights[x, y] = sr.ReadDouble();
                }
            }
            for (int i = 0; i < biases.Length; i++)
            {
                biases[i] = sr.ReadDouble();
            }
            sr.Close();
        }

        public bool IsMore() => pointer < files.Length;

        public void Erase()
        {
            foreach (string path in files) File.Delete(path);
        }

        public void WriteStructure(int[] structure)
        {
            BinaryWriter fs = new BinaryWriter(File.Create($@"{Address}\Structure.ai"));
            fs.Write(structure.Length);
            foreach (int i in structure) fs.Write(i);
            fs.Close();
        }

        public void Export(int index, double[,] weights, double[] biases)
        {
            BinaryWriter fs = new BinaryWriter(File.Create($@"{Address}\LAYER-{index}.ai"));
            for (int x = 0; x < weights.GetLength(0); x++)
            {
                for (int y = 0; y < weights.GetLength(1); y++)
                {
                    fs.Write(weights[x, y]);
                }
            }
            for (int x = 0; x < weights.GetLength(0); x++)
            {
                fs.Write(biases[x]);
            }
            fs.Close();
        }
    }
}
