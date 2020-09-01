using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Xml.Linq;

namespace AI_Framework
{
    public struct Entery
    {
        public readonly double[] input;
        public readonly double[] expected;

        public Entery (double[] input, double[] expected)
        {
            this.input = input;
            this.expected = expected;
        }

        public override string ToString()
        {
            string ret = "";
            for (int i = 0; i < input.Length; i++)
            {
                switch (input[i])
                {
                    case 0:
                        ret += ' ';
                        break;
                    case 1:
                        ret += '#';
                        break;
                    default:
                        ret += '.';
                        break;
                }
                if (i % 28 == 0) ret += '\n';
            }
            ret += $"\n{Array.IndexOf(expected, expected.Max())}";
            return ret;
        }
    }

    class Network_Pool
    {
        Neural_Network[] networks;
        Entery[] training_data;

        public Network_Pool(int count, int[] structure)
        {
            if (count % 2 != 0) count++;
            networks = new Neural_Network[count];
            for (int i = 0; i < count; i++)
            {
                networks[i] = new Neural_Network(structure);
            }
        }

        public void SetTrainingData(Entery[] training_data)
        {
            this.training_data = training_data;
        }

        public Neural_Network GetBest() => networks[0];

        public double Train(int sample_size, double chance, double max_delta)
        {
            ThreadedTrainer[] trainers = new ThreadedTrainer[networks.Length];
            Random random = new Random();
            Entery[] samples = new Entery[sample_size];

            for (int i = 0; i < sample_size; i++)
            {
                samples[i] = training_data[random.Next(training_data.Length)];
            }

            //samples = training_data;

            for (int i = 0; i < networks.Length; i++)
            {
                trainers[i] = new ThreadedTrainer(networks[i], samples);
            }

            foreach (ThreadedTrainer trainer in trainers)
            {
                trainer.Join();
                Console.WriteLine("trainer got a score of: {0}", trainer.GetCost());
            }

            trainers = trainers.OrderBy(x => x.GetCost()).ToArray();

            for (int i = 0; i < networks.Length / 2; i++)
            {
                networks[i * 2] = trainers[i].GetNetwork();
            }

            //Thread[] threads = new Thread[networks.Length / 2];

            for (int i = 0; i < networks.Length; i += 2)
            {
                networks[i + 1] = networks[i].Copy();
                //networks[i].Nudge(chance, max_delta);
                networks[i + 1].Nudge(chance, max_delta);
            }
            return trainers[0].GetCost();
        }

        private class ThreadedTrainer
        {
            private double ret = 0;
            public double correct = 0;
            private readonly Thread thread;
            private readonly Neural_Network network;
            private readonly Entery[] samples;

            public ThreadedTrainer(Neural_Network network, Entery[] samples)
            {
                this.network = network;
                this.samples = samples;
                thread = new Thread(new ThreadStart(Train));
                thread.Start();
            }

            public double GetCost() => ret;

            public Neural_Network GetNetwork() => network;

            public void Join()
            {
                thread.Join();
            }

            private void Train()
            {
                for (int i = 0; i < samples.Length; i++)
                {
                    network.ComputeNetwork(samples[i].input);
                    if (Array.IndexOf(network.GetLastResult(), network.GetLastResult().Max()) == Array.IndexOf(samples[i].expected, samples[i].expected.Max()))
                    {
                        correct++;
                    }
                    ret += network.ComputeCost(samples[i].expected) / samples.Length;
                }
            }
        }
    }
}
