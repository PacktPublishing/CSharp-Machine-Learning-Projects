using Accord.Controls;
using Accord.Math;
using Deedle;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Modeling
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.SetWindowSize(100, 60);

            // Read in the Cyber Attack dataset
            // TODO: change the path to point to your data directory
            string dataDirPath = @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.9\input-data";

            // Load the data into a data frame
            string dataPath = Path.Combine(dataDirPath, "pca-transformed-features.csv");
            Console.WriteLine("Loading {0}\n\n", dataPath);
            var featuresDF = Frame.ReadCsv(
                dataPath,
                hasHeaders: false,
                inferTypes: true
            );
            featuresDF.RenameColumns(
                featuresDF.ColumnKeys.Select((x, i) => String.Format("component-{0}", i + 1))
            );

            int[] labels = File.ReadLines(
                Path.Combine(dataDirPath, "pca-transformed-labels.csv")
            ).Select(x => int.Parse(x)).ToArray();
            featuresDF.AddColumn("attack_category", labels);

            Console.WriteLine("* Shape: ({0}, {1})\n\n", featuresDF.RowCount, featuresDF.ColumnCount);

            var count = featuresDF.AggregateRowsBy<string, int>(
                new string[] { "attack_category" },
                new string[] { "component-1" },
                x => x.ValueCount
            ).SortRows("component-1");
            count.RenameColumns(new string[] { "attack_category", "count" });
            count.Print();

            // First 13 components explain about 50% of the variance
            // First 19 components explain about 60% of the variance
            // First 27 components explain about 70% of the variance
            // First 34 components explain about 80% of the variance
            int numComponents = 27;
            string[] cols = featuresDF.ColumnKeys.Where((x, i) => i < numComponents).ToArray();

            // First, compute distances from the center/mean among normal events
            var normalDF = featuresDF.Rows[
                featuresDF["attack_category"].Where(x => x.Value == 0).Keys
            ].Columns[cols];

            double[][] normalData = BuildJaggedArray(
                normalDF.ToArray2D<double>(), normalDF.RowCount, cols.Length
            );
            double[] normalVariances = ComputeVariances(normalData);
            double[] rawDistances = ComputeDistances(normalData, normalVariances);

            // Filter out extreme values
            int[] idxFiltered = Matrix.ArgSort(rawDistances)
                .Where((x, i) =>  i < rawDistances.Length * 0.99).ToArray();
            double[] distances = rawDistances.Where((x, i) => idxFiltered.Contains(i)).ToArray();

            double meanDistance = distances.Average();
            double stdDistance = Math.Sqrt(
                distances
                .Select(x => Math.Pow(x - meanDistance, 2))
                .Sum() / distances.Length
            );

            Console.WriteLine(
                "\n\n* Normal - mean: {0:0.0000}, std: {1:0.0000}",
                meanDistance, stdDistance
            );

            HistogramBox.CheckForIllegalCrossThreadCalls = false;

            HistogramBox.Show(
                distances,
                title: "Distances"
            )
            .SetNumberOfBins(50);

            // Detection
            var attackDF = featuresDF.Rows[
                featuresDF["attack_category"].Where(x => x.Value > 0).Keys
            ].Columns[cols];

            double[][] attackData = BuildJaggedArray(
                attackDF.ToArray2D<double>(), attackDF.RowCount, cols.Length
            );
            double[] attackDistances = ComputeDistances(attackData, normalVariances);
            int[] attackLabels = featuresDF.Rows[
                featuresDF["attack_category"].Where(x => x.Value > 0).Keys
            ].GetColumn<int>("attack_category").ValuesAll.ToArray();

            // 5-10% false alarm rate
            for (int i = 4; i < 10; i++)
            {
                double targetFalseAlarmRate = 0.01 * (i + 1);
                double threshold = Accord.Statistics.Measures.Quantile(
                    distances,
                    1 - targetFalseAlarmRate
                );
                Console.WriteLine(threshold);
                int[] detected = attackDistances.Select(x => x > threshold ? 1 : 0).ToArray();

                EvaluateResults(attackLabels, detected, targetFalseAlarmRate);
            }

            Console.WriteLine("\n\n\n\n\nDONE!!!");
            Console.ReadKey();
        }

        private static void EvaluateResults(int[] attackLabels, int[] detected, double targetFalseAlarmRate)
        {
            double overallRecall = (double)detected.Sum() / attackLabels.Length;

            double[] truePositives = new double[4];
            double[] actualClassCounts = new double[4];

            for (int i = 0; i < attackLabels.Length; i++)
            {
                actualClassCounts[attackLabels[i] - 1] += 1.0;

                if (detected[i] > 0)
                {
                    truePositives[attackLabels[i] - 1] += 1.0;
                }
            }

            double[] recalls = truePositives.Select((x, i) => x / actualClassCounts[i]).ToArray();

            Console.WriteLine("\n\n---- {0:0.0}% False Alarm Rate ----", targetFalseAlarmRate * 100.0);
            Console.WriteLine("* Overall Attack Detection: {0:0.00}%", overallRecall * 100.0);
            Console.WriteLine(
                "* Detection by Attack Type:\n\t{0}",
                String.Join("\n\t", recalls.Select(
                    (x, i) => String.Format("Class {0}: {1:0.00}%", (i + 1), x * 100.0))
                )
            );
        }

        private static double[][] BuildJaggedArray(double[,] ary2d, int rowCount, int colCount)
        {
            double[][] matrix = new double[rowCount][];
            for (int i = 0; i < rowCount; i++)
            {
                matrix[i] = new double[colCount];
                for (int j = 0; j < colCount; j++)
                {
                    matrix[i][j] = double.IsNaN(ary2d[i, j]) ? 0.0 : ary2d[i, j];
                }
            }
            return matrix;
        }

        private static double[] ComputeVariances(double[][] data)
        {
            double[] componentVariances = new double[data[0].Length];

            for (int j = 0; j < data[0].Length; j++)
            {
                componentVariances[j] = data
                    .Select((x, i) => Math.Pow(data[i][j], 2))
                    .Sum() / data.Length;
            }

            return componentVariances;
        }

        private static double[] ComputeDistances(double[][] data, double[] componentVariances)
        {

            double[] distances = data.Select(
                (row, i) => Math.Sqrt(
                    row.Select(
                        (x, j) => Math.Pow(x, 2) / componentVariances[j]
                    ).Sum()
                )
            ).ToArray();

            return distances;
            
        }
    }
}
