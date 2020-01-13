using System;

namespace AI_Framework
{
    class Program
    {
        static void Main(string[] args)
        {
            Neural_Network network = new Neural_Network(new int[] { 2, 500, 1000, 500, 5 });
            Console.WriteLine("[{0}]", string.Join(", ", network.ComputeNetwork(new double[] { 0.6, 0.1 })));
            Console.WriteLine(network.ComputeCost(3));
            Console.ReadLine();
            network.Export(@"C:\Users\Nathan\Documents\AI");
            Console.ReadLine();
            network = new Neural_Network(@"C:\Users\Nathan\Documents\AI");
            Console.WriteLine("[{0}]", string.Join(", ", network.structure));
            Console.WriteLine("[{0}]", string.Join(", ", network.ComputeNetwork(new double[] { 0.6, 0.1 })));
            Console.WriteLine(network.ComputeCost(3));
        }
    }
}
