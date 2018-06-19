using Accord.Controls;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Analysis;
using Accord.Statistics.Kernels;
using Deedle;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ZedGraph;

namespace Modeling
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.SetWindowSize(100, 60);

            // Read in the Credit Card Fraud dataset
            // TODO: change the path to point to your data directory
            string dataDirPath = @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.10\input-data";

            // Load the data into a data frame
            string dataPath = Path.Combine(dataDirPath, "pca-features.csv");
            Console.WriteLine("Loading {0}\n\n", dataPath);
            var featuresDF = Frame.ReadCsv(
                dataPath,
                hasHeaders: false,
                inferTypes: true
            );
            featuresDF.RenameColumns(
                featuresDF.ColumnKeys
                    .Select((x, i) => i == featuresDF.ColumnCount - 1 ? "is_fraud" : String.Format("component-{0}", i + 1))
            );

            Console.WriteLine("* Shape: ({0}, {1})", featuresDF.RowCount, featuresDF.ColumnCount);

            var count = featuresDF.AggregateRowsBy<string, int>(
                new string[] { "is_fraud" },
                new string[] { "component-1" },
                x => x.ValueCount
            ).SortRows("component-1");
            count.RenameColumns(new string[] { "is_fraud", "count" });
            count.Print();

            // 1. Try PCA Classifier
            BuildPCAClassifier(featuresDF);
            // 2. One-Class SVM Classifier
            BuildOneClassSVM(featuresDF);


            Console.WriteLine("\n\n\n\n\nDONE!!!");
            Console.ReadKey();
        }

        private static void BuildOneClassSVM(Frame<int, string> featuresDF)
        {
            // First 13 components explain about 50% of the variance
            int numComponents = 13;
            string[] cols = featuresDF.ColumnKeys.Where((x, i) => i < numComponents).ToArray();

            var rnd = new Random(1);
            int[] trainIdx = featuresDF["is_fraud"]
                .Where(x => x.Value == 0)
                .Keys
                .OrderBy(x => rnd.Next())
                .Take(15000)
                .ToArray();
            var normalDF = featuresDF.Rows[
                trainIdx
            ].Columns[cols];

            //var normalDF = featuresDF.Columns[cols];

            double[][] normalData = BuildJaggedArray(
                normalDF.ToArray2D<double>(), normalDF.RowCount, cols.Length
            );

            var teacher = new OneclassSupportVectorLearning<Gaussian>();
            var model = teacher.Learn(normalData);

            int[] testIdx = featuresDF["is_fraud"]
                .Where(x => x.Value > 0)
                .Keys
                .Concat(
                    featuresDF["is_fraud"]
                    .Where(x => x.Value == 0 && !trainIdx.Contains(x.Key))
                    .Keys
                    .OrderBy(x => rnd.Next())
                    .Take(5000)
                    .ToArray()
                ).ToArray();

            var fraudDF = featuresDF.Rows[
                testIdx
            ].Columns[cols];

            double[][] fraudData = BuildJaggedArray(
                fraudDF.ToArray2D<double>(), fraudDF.RowCount, cols.Length
            );

            int[] fraudLabels = featuresDF.Rows[
                testIdx
            ].GetColumn<int>("is_fraud").ValuesAll.ToArray();

            for(int j = 0; j <= 10; j++)
            {
                model.Threshold = -1 + j/10.0; 

                int[] detected = new int[fraudData.Length];
                double[] probs = new double[fraudData.Length];
                for (int i = 0; i < fraudData.Length; i++)
                {
                    bool isNormal = model.Decide(fraudData[i]);
                    detected[i] = isNormal ? 0 : 1;
                }

                Console.WriteLine("\n\n---- One-Class SVM Results ----");
                Console.WriteLine("* Threshold: {0:0.00000}", model.Threshold);
                double correctPreds = fraudLabels
                    .Select((x, i) => detected[i] == 1 && x == 1 ? 1 : 0)
                    .Sum();
                double precision = correctPreds / detected.Sum();
                double overallRecall = correctPreds / fraudLabels.Sum();
                Console.WriteLine("* Overall Fraud Detection: {0:0.00}%", overallRecall * 100.0);
                Console.WriteLine("* Precision: {0:0.00}%", (precision) * 100.0);
                Console.WriteLine("* False Alarm Rate: {0:0.00}%", (1 - precision) * 100.0);
            }
        }

        private static void BuildPCAClassifier(Frame<int, string> featuresDF)
        {
            // First 13 components explain about 50% of the variance
            int numComponents = 13;
            string[] cols = featuresDF.ColumnKeys.Where((x, i) => i < numComponents).ToArray();

            // First, compute distances from the center/mean among normal events
            var normalDF = featuresDF.Rows[
                featuresDF["is_fraud"].Where(x => x.Value == 0).Keys
            ].Columns[cols];

            double[][] normalData = BuildJaggedArray(
                normalDF.ToArray2D<double>(), normalDF.RowCount, cols.Length
            );
            double[] normalVariances = ComputeVariances(normalData);
            double[] rawDistances = ComputeDistances(normalData, normalVariances);

            double[] distances = rawDistances.ToArray();

            double meanDistance = distances.Average();
            double stdDistance = Math.Sqrt(
                distances
                .Select(x => Math.Pow(x - meanDistance, 2))
                .Sum() / distances.Length
            );

            Console.WriteLine(
                "* Normal - mean: {0:0.0000}, std: {1:0.0000}",
                meanDistance, stdDistance
            );

            // Detection
            var fraudDF = featuresDF.Rows[
                featuresDF["is_fraud"].Where(x => x.Value > 0).Keys
            ].Columns[cols];

            double[][] fraudData = BuildJaggedArray(
                fraudDF.ToArray2D<double>(), fraudDF.RowCount, cols.Length
            );
            double[] fraudDistances = ComputeDistances(fraudData, normalVariances);
            int[] fraudLabels = featuresDF.Rows[
                featuresDF["is_fraud"].Where(x => x.Value > 0).Keys
            ].GetColumn<int>("is_fraud").ValuesAll.ToArray();

            // 5-10% false alarm rate
            for (int i = 0; i < 4; i++)
            {
                double targetFalseAlarmRate = 0.05 * (i + 1);
                double threshold = Accord.Statistics.Measures.Quantile(
                    distances,
                    1 - targetFalseAlarmRate
                );

                int[] detected = fraudDistances.Select(x => x > threshold ? 1 : 0).ToArray();

                Console.WriteLine("\n\n---- {0:0.0}% False Alarm Rate ----", targetFalseAlarmRate * 100.0);
                double overallRecall = (double)detected.Sum() / detected.Length;
                Console.WriteLine("* Overall Fraud Detection: {0:0.00}%", overallRecall * 100.0);
            }
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
