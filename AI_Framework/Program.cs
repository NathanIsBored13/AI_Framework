using AI_Framework;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;

namespace ReadMNIST
{
    class Program
    {
        private static Network_Pool networks = new Network_Pool(50, new int[] { 784, 20, 20, 20, 10 });
        private static bool ThreadStop;

        static void Main()
        {
            //Entery[] data = new Entery[4];
            //for (int i = 0; i < 4; i++)
            //{
            //    data[i] = output((int)Math.Floor(i / 2d), i % 2);
            //}
            Entery[] data = Load_Data();
            networks.SetTrainingData(data);
            string[] input;
            do
            {
                Console.Write("Enter Command\n>");
                input = Console.ReadLine().Split(' ');
                Random random = new Random();
                switch (input[0].ToLower())
                {
                    case "train":
                        ThreadStop = false;
                        Thread training = new Thread(new ThreadStart(Train));
                        training.Start();
                        Console.ReadKey();
                        ThreadStop = true;
                        training.Join();
                        break;
                    case "sample":
                        Entery ent = data[random.Next(data.Length)];
                        Console.WriteLine(ent.ToString());
                        double[] output = networks.GetBest().ComputeNetwork(ent.input);
                        Console.WriteLine($"[{string.Join(", ", output)}]");
                        Console.WriteLine($"The network predicted: {Array.IndexOf(output, output.Max())}");
                        Console.WriteLine(networks.GetBest().ComputeCost(ent.expected));
                        break;
                }
            } while (input[0].ToLower() != "quit");
        }

        private static void Train()
        {
            while (!ThreadStop)
            {
                Console.WriteLine(networks.Train(500, 0.01, 0.5));
            }
        }

        public static Entery output(int a, int b)
        {
            return new Entery(new double[] { a, b }, new double[] { Exor(a, b)? 0 : 1, Exor(a, b)? 1 : 0});
        }

        public static bool Exor(int a, int b) => a != b;

        private static Entery[] Load_Data()
        {
            BinaryReader brLabels = new BinaryReader(new FileStream(@"C:\t10k-labels.idx1-ubyte", FileMode.Open));
            BinaryReader brImages = new BinaryReader(new FileStream(@"C:\t10k-images.idx3-ubyte", FileMode.Open));

            brImages.ReadBytes(4);
            brLabels.ReadBytes(8);

            int numImages = BtoLEndian(brImages.ReadBytes(4));
            int numRows = BtoLEndian(brImages.ReadBytes(4));
            int numCols = BtoLEndian(brImages.ReadBytes(4));

            Entery[] ret = new Entery[numImages];
            for (int i = 0; i < numImages; i++)
            {
                double[] input = new double[numRows * numCols];
                double[] expected = new double[10];
                for (int j = 0; j < numRows * numCols; j++)
                {
                    input[j] = brImages.ReadByte() / 256d;
                }
                expected[brLabels.ReadByte()] = 1;
                ret[i] = new Entery(input, expected);
            }
            brImages.Close();
            brLabels.Close();
            return ret;
        }

        private static int BtoLEndian(byte[] i)
        {
            Array.Reverse(i);
            return BitConverter.ToInt32(i, 0);
        }
    }
}
