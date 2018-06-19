using Accord.Controls;
using Deedle;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataAnalyzer
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.SetWindowSize(100, 55);

            // Read in the Credit Card Fraud dataset
            // TODO: change the path to point to your data directory
            string dataDirPath = @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.10\input-data";

            // Load the data into a data frame
            string dataPath = Path.Combine(dataDirPath, "creditcard.csv");
            Console.WriteLine("Loading {0}\n\n", dataPath);
            var df = Frame.ReadCsv(
                dataPath,
                hasHeaders: true,
                inferTypes: true
            );

            Console.WriteLine("* Shape: {0}, {1}\n\n", df.RowCount, df.ColumnCount);

            // Target variable distribution
            var targetVarCount = df.AggregateRowsBy<string, int>(
                new string[] { "Class" },
                new string[] { "V1" },
                x => x.ValueCount
            ).SortRows("V1");
            targetVarCount.RenameColumns(new string[] { "is_fraud", "count" });

            targetVarCount.Print();

            DataBarBox.Show(
                targetVarCount.GetColumn<string>("is_fraud").Values.ToArray(),
                targetVarCount["count"].Values.ToArray()
            ).SetTitle(
                "Counts by Target Class"
            );

            // Feature distributions
            HistogramBox.CheckForIllegalCrossThreadCalls = false;

            foreach (string col in df.ColumnKeys)
            {
                if (col.Equals("Class") || col.Equals("Time"))
                {
                    continue;
                }

                double[] values = df[col].DropMissing().ValuesAll.ToArray();
                // Compute Quartiles
                Console.WriteLine(String.Format("\n\n-- {0} Distribution -- ", col));
                double[] quartiles = Accord.Statistics.Measures.Quantiles(
                    values,
                    new double[] { 0, 0.25, 0.5, 0.75, 1.0 }
                );
                Console.WriteLine(
                    "Min: \t\t\t{0:0.00}\nQ1 (25% Percentile): \t{1:0.00}\nQ2 (Median): \t\t{2:0.00}\nQ3 (75% Percentile): \t{3:0.00}\nMax: \t\t\t{4:0.00}",
                    quartiles[0], quartiles[1], quartiles[2], quartiles[3], quartiles[4]
                );
                // Visualize Distributions
                HistogramBox.Show(
                    values,
                    title: col
                )
                .SetNumberOfBins(50);
            }

            // Target Var Distributions on 2-dimensional feature space
            double[][] data = BuildJaggedArray(
                df.ToArray2D<double>(), df.RowCount, df.ColumnCount
            );
            int[] labels = df.GetColumn<int>("Class").ValuesAll.ToArray();

            double[][] first2Components = data.Select(
                x => x.Where((y, i) => i < 2
            ).ToArray()).ToArray();
            ScatterplotBox.Show("Feature #1 vs. Feature #2", first2Components, labels);

            double[][] next2Components = data.Select(
                x => x.Where((y, i) => i >= 1 && i <= 2).ToArray()
            ).ToArray();
            ScatterplotBox.Show("Feature #2 vs. Feature #3", next2Components, labels);

            next2Components = data.Select(
                x => x.Where((y, i) => i >= 2 && i <= 3).ToArray()
            ).ToArray();
            ScatterplotBox.Show("Feature #3 vs. Feature #4", next2Components, labels);

            Console.WriteLine("\n\n\n\n\nDONE!!!");
            Console.ReadKey();
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
    }
}
