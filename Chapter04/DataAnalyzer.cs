using Accord.Controls;
using Deedle;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace DataAnalyzer
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.SetWindowSize(100, 50);

            // Read in the OHLC dataset
            // TODO: change the path to point to your data directory
            string dataDirPath = @"<path-to-your-data-dir>";

            // Load the OHLC data into a data frame
            string ohlcDataPath = Path.Combine(dataDirPath, "eurusd-daily-ohlc.csv");
            Console.WriteLine("Loading {0}\n", ohlcDataPath);
            var ohlcDF = Frame.ReadCsv(
                ohlcDataPath,
                hasHeaders: true,
                inferTypes: true
            );

            // Time-series line chart of close prices
            var closePriceLineChart = DataSeriesBox.Show(
                ohlcDF.RowKeys.Select(x => (double)x),
                ohlcDF.GetColumn<double>("Close").ValuesAll
            );

            System.Threading.Thread.Sleep(3000);
            closePriceLineChart.Invoke(
                new Action(() =>
                {
                    closePriceLineChart.Size = new System.Drawing.Size(700, 500);
                })
            );

            // Time-series line chart of daily returns
            var dailyReturnLineChart = DataSeriesBox.Show(
                ohlcDF.RowKeys.Select(x => (double)x),
                ohlcDF.FillMissing(0.0)["DailyReturn"].ValuesAll
            );

            System.Threading.Thread.Sleep(3000);
            dailyReturnLineChart.Invoke(
                new Action(() =>
                {
                    dailyReturnLineChart.Size = new System.Drawing.Size(700, 500);
                })
            );

            var dailyReturnHistogram = HistogramBox
            .Show(
                ohlcDF.FillMissing(0.0)["DailyReturn"].ValuesAll.ToArray()
            )
            .SetNumberOfBins(20);

            System.Threading.Thread.Sleep(3000);
            dailyReturnHistogram.Invoke(
                new Action(() =>
                {
                    dailyReturnHistogram.Size = new System.Drawing.Size(700, 500);
                })
            );

            // Check the distribution of daily returns
            double returnMax = ohlcDF["DailyReturn"].Max();
            double returnMean = ohlcDF["DailyReturn"].Mean();
            double returnMedian = ohlcDF["DailyReturn"].Median();
            double returnMin = ohlcDF["DailyReturn"].Min();
            double returnStdDev = ohlcDF["DailyReturn"].StdDev();

            double[] quantiles = Accord.Statistics.Measures.Quantiles(
                ohlcDF.FillMissing(0.0)["DailyReturn"].ValuesAll.ToArray(),
                new double[] { 0.25, 0.5, 0.75 }
            );

            Console.WriteLine("-- DailyReturn Distribution-- ");

            Console.WriteLine("Mean: \t\t\t{0:0.00}\nStdDev: \t\t{1:0.00}\n", returnMean, returnStdDev);

            Console.WriteLine(
                "Min: \t\t\t{0:0.00}\nQ1 (25% Percentile): \t{1:0.00}\nQ2 (Median): \t\t{2:0.00}\nQ3 (75% Percentile): \t{3:0.00}\nMax: \t\t\t{4:0.00}",
                returnMin, quantiles[0], quantiles[1], quantiles[2], returnMax
            );

            Console.WriteLine("\nDONE!!!");
            Console.ReadKey();
        }
    }
}
