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
            Console.SetWindowSize(100, 60);

            // Read in the Audio Features dataset
            // TODO: change the path to point to your data directory
            string dataDirPath = @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.7\input-data";

            // Load the data into a data frame
            string dataPath = Path.Combine(dataDirPath, "sample.csv");
            Console.WriteLine("Loading {0}\n\n", dataPath);
            var featuresDF = Frame.ReadCsv(
                dataPath,
                hasHeaders: true,
                inferTypes: true
            );

            Console.WriteLine("* Shape: {0}, {1}\n\n", featuresDF.RowCount, featuresDF.ColumnCount);

            var genreCount = featuresDF.AggregateRowsBy<string, int>(
                new string[] { "genre_top" },
                new string[] { "track_id" },
                x => x.ValueCount
            ).SortRows("track_id");

            genreCount.Print();

            var barChart = DataBarBox.Show(
                genreCount.GetColumn<string>("genre_top").Values.ToArray().Select(x => x.Substring(0,3)),
                genreCount["track_id"].Values.ToArray()
            ).SetTitle(
                "Genre Count"
            );

            foreach (string col in featuresDF.ColumnKeys)
            {
                if (col.StartsWith("mfcc"))
                {
                    int idx = int.Parse(col.Split('.')[2]);
                    if(idx <= 4)
                    {
                        Console.WriteLine(String.Format("\n\n-- {0} Distribution -- ", col));
                        double[] quantiles = Accord.Statistics.Measures.Quantiles(
                            featuresDF[col].ValuesAll.ToArray(),
                            new double[] { 0, 0.25, 0.5, 0.75, 1.0 }
                        );
                        Console.WriteLine(
                            "Min: \t\t\t{0:0.00}\nQ1 (25% Percentile): \t{1:0.00}\nQ2 (Median): \t\t{2:0.00}\nQ3 (75% Percentile): \t{3:0.00}\nMax: \t\t\t{4:0.00}",
                            quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4]
                        );
                    }
                }
            }

            string[] attributes = new string[] { "kurtosis", "min", "max", "mean", "median", "skew", "std" };
            foreach (string attribute in attributes)
            {
                string[] featureColumns = featuresDF.ColumnKeys.Where(x => x.Contains(attribute)).ToArray();
                foreach (string genre in genreCount.GetColumn<string>("genre_top").Values)
                {
                    var genreDF = featuresDF.Rows[
                        featuresDF.GetColumn<string>("genre_top").Where(x => x.Value == genre).Keys
                    ].Columns[featureColumns];

                    ScatterplotBox.Show(
                        BuildXYPairs(
                            genreDF.Columns[featureColumns].ToArray2D<double>(),
                            genreDF.RowCount,
                            genreDF.ColumnCount
                        )
                    ).SetTitle(String.Format("{0}-{1}", genre, attribute));
                }
            }


            Console.WriteLine("\n\n\n\n\nDONE!!!");
            Console.ReadKey();
        }


        private static double[][] BuildXYPairs(double[,] ary2D, int rowCount, int columnCount)
        {
            double[][] ary = new double[rowCount*columnCount][];
            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < columnCount; j++)
                {
                    ary[i * columnCount + j] = new double[2];
                    ary[i * columnCount + j][0] = j + 1;
                    ary[i * columnCount + j][1] = ary2D[i, j];
                }
            }
            return ary;
        }
    }
}
