using Accord.Controls;
using Accord.Statistics.Analysis;
using Deedle;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FeatureEngineering
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
            string dataPath = Path.Combine(dataDirPath, "creditcard.csv");
            Console.WriteLine("Loading {0}\n\n", dataPath);
            var df = Frame.ReadCsv(
                dataPath,
                hasHeaders: true,
                inferTypes: true
            );

            Console.WriteLine("* Shape: {0}, {1}\n\n", df.RowCount, df.ColumnCount);

            string[] featureCols = df.ColumnKeys.Where(
                x => !x.Equals("Time") && !x.Equals("Class")
            ).ToArray();

            var noFraudData = df.Rows[
                df["Class"].Where(x => x.Value == 0.0).Keys
            ].Columns[featureCols];
            double[][] data = BuildJaggedArray(
                noFraudData.ToArray2D<double>(), noFraudData.RowCount, featureCols.Length
            );

            double[][] wholeData = BuildJaggedArray(
                df.Columns[featureCols].ToArray2D<double>(), df.RowCount, featureCols.Length
            );
            int[] labels = df.GetColumn<int>("Class").ValuesAll.ToArray();

            var pca = new PrincipalComponentAnalysis(
                PrincipalComponentMethod.Standardize
            );
            pca.Learn(data);

            double[][] transformed = pca.Transform(wholeData);
            double[][] first2Components = transformed.Select(x => x.Where((y, i) => i < 2).ToArray()).ToArray();
            ScatterplotBox.Show("Component #1 vs. Component #2", first2Components, labels);
            double[][] next2Components = transformed.Select(
                x => x.Where((y, i) => i >= 1 && i <= 2).ToArray()
            ).ToArray();
            ScatterplotBox.Show("Component #2 vs. Component #3", next2Components, labels);
            next2Components = transformed.Select(
                x => x.Where((y, i) => i >= 2 && i <= 3).ToArray()
            ).ToArray();
            ScatterplotBox.Show("Component #3 vs. Component #4", next2Components, labels);
            next2Components = transformed.Select(
                x => x.Where((y, i) => i >= 3 && i <= 4).ToArray()
            ).ToArray();
            ScatterplotBox.Show("Component #4 vs. Component #5", next2Components, labels);

            DataSeriesBox.Show(
                pca.Components.Select((x, i) => (double)i),
                pca.Components.Select(x => x.CumulativeProportion)
            ).SetTitle("Explained Variance");
            System.IO.File.WriteAllLines(
                Path.Combine(dataDirPath, "explained-variance.csv"),
                pca.Components.Select((x, i) => String.Format("{0},{1:0.0000}", i + 1, x.CumulativeProportion))
            );

            Console.WriteLine("exporting train set...");

            System.IO.File.WriteAllLines(
                Path.Combine(dataDirPath, "pca-features.csv"),
                transformed.Select((x, i) => String.Format("{0},{1}", String.Join(",", x), labels[i]))
            );


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
