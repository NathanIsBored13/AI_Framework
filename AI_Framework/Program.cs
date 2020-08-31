using AI_Framework;
using System;
using System.IO;
using System.Linq;

namespace ReadMNIST
{
    class Program
    {
        static void Main()
        {
            Network_Pool networks = new Network_Pool(20, new int[] { 256, 20, 20, 10 });
            networks.SetTrainingData(Load_Data());
            Console.WriteLine(networks.Train(10));
        }

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
