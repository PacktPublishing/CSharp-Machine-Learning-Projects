using Accord.Controls;
using Deedle;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
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

            ISet<string> exportedLabels = new HashSet<string>();
            for(int i = 0; i < featuresDF.RowCount; i++)
            {
                exportedLabels.Add(featuresDF.Rows[i].GetAs<string>("label"));

                CreateImage(
                    featuresDF.Rows[i].ValuesAll.Select(x => (int)x).Where((x, idx) => idx > 0).ToArray(),
                    featuresDF.Rows[i].GetAs<string>("label")
                );

                if(exportedLabels.Count() >= 10)
                {
                    break;
                }
            }

            var digitCount = featuresDF.AggregateRowsBy<string, int>(
                new string[] { "label" },
                new string[] { "pixel0" },
                x => x.ValueCount
            ).SortRows("pixel0");

            digitCount.Print();

            var barChart = DataBarBox.Show(
                digitCount.GetColumn<string>("label").Values.ToArray(),
                digitCount["pixel0"].Values.ToArray()
            ).SetTitle(
                "Digit Count"
            );

            List<string> featureCols = new List<string>();
            foreach (string col in featuresDF.ColumnKeys)
            {
                if (featureCols.Count >= 20)
                {
                    break;
                }

                if (col.StartsWith("pixel"))
                {
                    if (featuresDF[col].Max() > 0)
                    {
                        featureCols.Add(col);

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

            string[] featureColumns = featureCols.ToArray();

            foreach (string label in digitCount.GetColumn<string>("label").Values)
            {
                var subfeaturesDF = featuresDF.Rows[
                    featuresDF.GetColumn<string>("label").Where(x => x.Value == label).Keys
                ].Columns[featureColumns];

                ScatterplotBox.Show(
                    BuildXYPairs(
                        subfeaturesDF.Columns[featureColumns].ToArray2D<double>(),
                        subfeaturesDF.RowCount,
                        subfeaturesDF.ColumnCount
                    )
                ).SetTitle(String.Format("Digit: {0} - 20 sample Pixels", label));
            }

            double[][] twoPixels = featuresDF.Columns[
                new string[] { featureColumns[15], featureColumns[16] }
            ].Rows.Select(
                x => Array.ConvertAll<object, double>(x.Value.ValuesAll.ToArray(), o => Convert.ToDouble(o))
            ).ValuesAll.ToArray();

            ScatterplotBox.Show(
                String.Format("{0} vs. {1}", featureColumns[15], featureColumns[16]), 
                twoPixels,
                featuresDF.GetColumn<int>("label").Values.ToArray()
            );

            Console.WriteLine("\n\n\n\n\nDONE!!!");
            Console.ReadKey();
        }

        private static double[][] BuildXYPairs(double[,] ary2D, int rowCount, int columnCount)
        {
            double[][] ary = new double[rowCount * columnCount][];
            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < columnCount; j++)
                {
                    ary[i * columnCount + j] = new double[2];
                    ary[i * columnCount + j][0] = j;
                    ary[i * columnCount + j][1] = ary2D[i, j];
                }
            }
            return ary;
        }

        private static void CreateImage(int[] rows, string digit)
        {
            int width = 28;
            int height = 28;
            int stride = width * 4;
            int[,] pixelData = new int[width, height];

            for (int i = 0; i < width; ++i)
            {
                for (int j = 0; j < height; ++j)
                {
                    byte[] bgra = new byte[] { (byte)rows[28 * i + j], (byte)rows[28 * i + j], (byte)rows[28 * i + j], 255 };
                    pixelData[i, j] = BitConverter.ToInt32(bgra, 0);
                }
            }

            Bitmap bitmap;
            unsafe
            {
                fixed (int* ptr = &pixelData[0, 0])
                {
                    bitmap = new Bitmap(width, height, stride, PixelFormat.Format32bppRgb, new IntPtr(ptr));
                }
            }
            bitmap.Save(
                String.Format(@"\\Mac\Home\Documents\c-sharp-machine-learning\ch.8\input-data\{0}.jpg", digit)
            );
        }
    }
}
