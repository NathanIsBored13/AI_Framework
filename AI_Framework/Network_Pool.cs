using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading;

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
    }

    class Network_Pool
    {
        Neural_Network[] networks;
        Entery[] training_data;
        public Network_Pool(int count, int[] structure)
        {
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

        public double Train(int sample_size)
        {
            ThreadedTrainer[] trainers = new ThreadedTrainer[networks.Length];
            double[] score = new double[networks.Length];
            Random random = new Random();
            Entery[] samples = new Entery[sample_size];
            for (int i = 0; i < sample_size; i++)
            {
                samples[i] = training_data[random.Next(training_data.Length)];
            }
            for (int i = 0; i < networks.Length; i++)
            {
                Console.WriteLine($"thread {i} started");
                trainers[i] = new ThreadedTrainer(networks[i], samples);
            }

            for (int i = 0; i < networks.Length; i++)
            {
                score[i] = trainers[i].Join();
                Console.WriteLine($"thread {i} stopped with score {score[i]}");
            }

            return score.Min();
        }

        private class ThreadedTrainer
        {
            private double ret = 0;
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

            public double Join()
            {
                thread.Join();
                return ret;
            }

            private void Train()
            {
                for (int i = 0; i < samples.Length; i++)
                {
                    network.ComputeNetwork(samples[i].input);
                    ret += network.ComputeCost(samples[i].expected) / samples.Length;
                }
            }
        }
    }
}
