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

            // Read in the Image Features dataset
            // TODO: change the path to point to your data directory
            string dataDirPath = @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.8\input-data";

            // Load the data into a data frame
            string dataPath = Path.Combine(dataDirPath, "train.csv");
            Console.WriteLine("Loading {0}\n\n", dataPath);
            var featuresDF = Frame.ReadCsv(
                dataPath,
                hasHeaders: true,
                inferTypes: true
            );

            Console.WriteLine("* Shape: {0}, {1}\n\n", featuresDF.RowCount, featuresDF.ColumnCount);

            double trainSetProportiona = 0.7;

            var rnd = new Random();
            var trainIdx = featuresDF.RowKeys.Where((x, i) => rnd.NextDouble() <= trainSetProportiona);
            var testIdx = featuresDF.RowKeys.Where((x, i) => !trainIdx.Contains(i));

            var trainset = featuresDF.Rows[trainIdx];
            var testset = featuresDF.Rows[testIdx];

            var trainLabels = trainset.GetColumn<int>("label").Values.ToArray();

            string[] nonZeroPixelCols = trainset.ColumnKeys.Where(x => trainset[x].Max() > 0 && !x.Equals("label")).ToArray();

            double[][] data = trainset.Columns[nonZeroPixelCols].Rows.Select(
                x => Array.ConvertAll<object, double>(x.Value.ValuesAll.ToArray(), o => Convert.ToDouble(o))
            ).ValuesAll.ToArray();

            Console.WriteLine("* Shape: {0}, {1}\n\n", data.Length, data[0].Length);

            var digitCount = trainset.AggregateRowsBy<string, int>(
                new string[] { "label" },
                new string[] { "pixel0" },
                x => x.ValueCount
            ).SortRows("pixel0");

            digitCount.Print();

            var barChart = DataBarBox.Show(
                digitCount.GetColumn<string>("label").Values.ToArray(),
                digitCount["pixel0"].Values.ToArray()
            ).SetTitle(
                "Train Set - Digit Count"
            );

            digitCount = testset.AggregateRowsBy<string, int>(
                new string[] { "label" },
                new string[] { "pixel0" },
                x => x.ValueCount
            ).SortRows("pixel0");

            digitCount.Print();

            barChart = DataBarBox.Show(
                digitCount.GetColumn<string>("label").Values.ToArray(),
                digitCount["pixel0"].Values.ToArray()
            ).SetTitle(
                "Test Set - Digit Count"
            );

            var pca = new PrincipalComponentAnalysis(
                PrincipalComponentMethod.Standardize
            );
            pca.Learn(data);

            double[][] transformed = pca.Transform(data);
            double[][] first2Components = transformed.Select(x => x.Where((y, i) => i < 2).ToArray()).ToArray();
            ScatterplotBox.Show("Component #1 vs. Component #2", first2Components, trainLabels);

            DataSeriesBox.Show(
                pca.Components.Select((x, i) => (double)i),
                pca.Components.Select(x => x.CumulativeProportion)
            ).SetTitle("Explained Variance");
            System.IO.File.WriteAllLines(
                Path.Combine(dataDirPath, "explained-variance.csv"),
                pca.Components.Select((x, i) => String.Format("{0},{1:0.0000}", i, x.CumulativeProportion))
            );

            Console.WriteLine("exporting train set...");
            var trainTransformed = pca.Transform(
                trainset.Columns[nonZeroPixelCols].Rows.Select(
                    x => Array.ConvertAll<object, double>(x.Value.ValuesAll.ToArray(), o => Convert.ToDouble(o))
                ).ValuesAll.ToArray()
            );

            System.IO.File.WriteAllLines(
                Path.Combine(dataDirPath, "pca-train.csv"),
                trainTransformed.Select((x, i) => String.Format("{0},{1}", String.Join(",", x), trainset["label"].GetAt(i)))
            );

            Console.WriteLine("exporting test set...");
            var testTransformed = pca.Transform(
                testset.Columns[nonZeroPixelCols].Rows.Select(
                    x => Array.ConvertAll<object, double>(x.Value.ValuesAll.ToArray(), o => Convert.ToDouble(o))
                ).ValuesAll.ToArray()
            );
            System.IO.File.WriteAllLines(
                Path.Combine(dataDirPath, "pca-test.csv"),
                testTransformed.Select((x, i) => String.Format("{0},{1}", String.Join(",", x), testset["label"].GetAt(i)))
            );

            Console.WriteLine("\n\n\n\n\nDONE!!!");
            Console.ReadKey();
        }
    }
}
