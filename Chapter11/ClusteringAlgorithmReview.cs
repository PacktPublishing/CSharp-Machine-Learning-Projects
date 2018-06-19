using Accord.MachineLearning;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ClusteringAlgorithmReview
{
    class Program
    {
        static void Main(string[] args)
        {
            // sample input
            var sampleSet = new double[][]
            {
                new double[] { 1, 9 },
                new double[] { 2, 8 },
                new double[] { 3, 7 },
                new double[] { 4, 6 },
                new double[] { 5, 5 }
            };

            KMeans kmeans = new KMeans(2);
            KMeansClusterCollection clusters = kmeans.Learn(sampleSet);

            Console.WriteLine("\n\n* Clusters: {0}", String.Join(",", clusters.Decide(sampleSet)));

            Console.WriteLine("\n\n\n\nDONE!!");
            Console.ReadKey();

        }
    }
}
