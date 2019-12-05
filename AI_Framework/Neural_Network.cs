using System;
using System.Collections.Generic;
using System.Text;

namespace AI_Framework
{
    class Neural_Network
    {
        public int[] structure { get; }
        private Layer input;
        public Neural_Network(int[] structure)
        {
            this.structure = structure;
            input = new Layer(null, structure, 1);
        }
        public double[] Compute_Network(double[] input) => this.input.Propegate_Down(input);
    }
}
