﻿using System;
using System.Text;
using System.IO;

namespace AI_Framework
{
    class File_Maneger
    {
        private string address;
        private int pointer = 0;
        private string[] files;
        public File_Maneger(string address)
        {
            this.address = address;
            files = Directory.GetFiles(address, @"*.csv");
        }
        public int[] GetStructure()
        {
            int[] ret = new int[files.Length + 1];
            StreamReader sr = new StreamReader(files[0]);
            ret[0] = Convert.ToInt32(sr.ReadLine().Split(", ")[1]);
            sr.Close();
            for (int i = 0; i < files.Length; i++)
            {
                sr = new StreamReader(files[i]);
                ret[i + 1] = Convert.ToInt32(sr.ReadLine().Split(", ")[0]);
                sr.Close();
            }
            return ret;
        }
        public LayerInfo ReadNext()
        {
            Console.WriteLine("read {0}", files[pointer]);
            StreamReader sr = new StreamReader(files[pointer++]);
            string[] buffer = sr.ReadLine().Split(", ");
            double[,] weights = new double[Convert.ToInt32(buffer[0]), Convert.ToInt32(buffer[1])];
            double[] biases = new double[Convert.ToInt32(buffer[0])];
            for (int x = 0; x < weights.GetLength(0); x++)
            {
                buffer = sr.ReadLine().Split(",");
                for (int y = 0; y < weights.GetLength(1); y++)
                {
                    weights[x, y] = Convert.ToDouble(buffer[y]);
                }
            }
            buffer = sr.ReadLine().Split(",");
            for (int i = 0; i < buffer.Length; i++)
            {
                biases[i] = Convert.ToDouble(buffer[i]);
            }
            sr.Close();
            return new LayerInfo
            {
                weights = weights,
                biases = biases
            };
        }
        public bool IsMore() => pointer < files.Length;
        public void Erase()
        {
            foreach (string path in files) File.Delete(path);
        }
        public void Export(int index, double[,] weights, double[] biases)
        {
            FileStream fs = File.Create(string.Format(@"{0}\LAYER-{1}.csv", address, index));
            Write(fs, string.Format("{0},{1}\n", weights.GetLength(0), weights.GetLength(1)));
            for (int i = 0; i < weights.GetLength(0); i++) Write(fs, string.Format("{0}\n", string.Join(",", GetRow(weights, i))));
            Write(fs, string.Join(",", biases));
            fs.Close();
        }
        private void Write(FileStream fs, string value)
        {
            byte[] write = new UTF8Encoding(true).GetBytes(value);
            fs.Write(write, 0, write.Length);
        }
        private double[] GetRow(double[,] matrix, int index)
        {
            double[] ret = new double[matrix.GetLength(1)];
            for (int i = 0; i < ret.Length; i++) ret[i] = matrix[index, i];
            return ret;
        }
    }
}