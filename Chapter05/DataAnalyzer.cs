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
            Console.SetWindowSize(100, 50);

            // Read in the House Price dataset
            // TODO: change the path to point to your data directory
            string dataDirPath = @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.5\input-data";

            // Load the data into a data frame
            string dataPath = Path.Combine(dataDirPath, "train.csv");
            Console.WriteLine("Loading {0}\n", dataPath);
            var houseDF = Frame.ReadCsv(
                dataPath,
                hasHeaders: true,
                inferTypes: true
            );

            // Categorical Variable #1: Building Type
            Console.WriteLine("\nCategorical Variable #1: Building Type");
            var buildingTypeDistribution = houseDF.GetColumn<string>(
                "BldgType"
            ).GroupBy<string>(x => x.Value).Select(x => (double)x.Value.KeyCount);
            buildingTypeDistribution.Print();

            var buildingTypeBarChart = DataBarBox.Show(
                buildingTypeDistribution.Keys.ToArray(),
                buildingTypeDistribution.Values.ToArray()
            );
            buildingTypeBarChart.SetTitle("Building Type Distribution (Categorical)");
            System.Threading.Thread.Sleep(3000);
            buildingTypeBarChart.Invoke(
                new Action(() =>
                {
                    buildingTypeBarChart.Size = new System.Drawing.Size(1000, 700);
                })
            );

            // Categorical Variable #2: Lot Configuration
            Console.WriteLine("\nCategorical Variable #1: Building Type");
            var lotConfigDistribution = houseDF.GetColumn<string>(
                "LotConfig"
            ).GroupBy<string>(x => x.Value).Select(x => (double)x.Value.KeyCount);
            lotConfigDistribution.Print();

            var lotConfigBarChart = DataBarBox.Show(
                lotConfigDistribution.Keys.ToArray(),
                lotConfigDistribution.Values.ToArray()
            );
            lotConfigBarChart.SetTitle("Lot Configuration Distribution (Categorical)");
            System.Threading.Thread.Sleep(3000);
            lotConfigBarChart.Invoke(
                new Action(() =>
                {
                    lotConfigBarChart.Size = new System.Drawing.Size(1000, 700);
                })
            );

            // Ordinal Categorical Variable #1: Overall material and finish of the house
            Console.WriteLine("\nOrdinal Categorical #1: Overall material and finish of the house");
            var overallQualDistribution = houseDF.GetColumn<string>(
                "OverallQual"
            ).GroupBy<int>(
                x => Convert.ToInt32(x.Value)
            ).Select(
                x => (double)x.Value.KeyCount
            ).SortByKey().Reversed;
            overallQualDistribution.Print();

            var overallQualBarChart = DataBarBox.Show(
                overallQualDistribution.Keys.Select(x => x.ToString()),
                overallQualDistribution.Values.ToArray()
            );
            overallQualBarChart.SetTitle("Overall House Quality Distribution (Ordinal)");
            System.Threading.Thread.Sleep(3000);
            overallQualBarChart.Invoke(
                new Action(() =>
                {
                    overallQualBarChart.Size = new System.Drawing.Size(1000, 700);
                })
            );

            // Ordinal Categorical Variable #2: Exterior Quality
            Console.WriteLine("\nOrdinal Categorical #2: Exterior Quality");
            var exteriorQualDistribution = houseDF.GetColumn<string>(
                "ExterQual"
            ).GroupBy<string>(x => x.Value).Select(
                x => (double)x.Value.KeyCount
            )[new string[] { "Ex", "Gd", "TA", "Fa" }];
            exteriorQualDistribution.Print();

            var exteriorQualBarChart = DataBarBox.Show(
                exteriorQualDistribution.Keys.Select(x => x.ToString()),
                exteriorQualDistribution.Values.ToArray()
            );
            exteriorQualBarChart.SetTitle("Exterior Quality Distribution (Ordinal)");
            System.Threading.Thread.Sleep(3000);
            exteriorQualBarChart.Invoke(
                new Action(() =>
                {
                    exteriorQualBarChart.Size = new System.Drawing.Size(1000, 700);
                })
            );

            HistogramBox.CheckForIllegalCrossThreadCalls = false;

            // Continuous Variable #1-1: First Floor Square Feet
            var firstFloorHistogram = HistogramBox
            .Show(
                houseDF.DropSparseRows()["1stFlrSF"].ValuesAll.ToArray(),
                title: "First Floor Square Feet (Continuous)"
            )
            .SetNumberOfBins(20);

            System.Threading.Thread.Sleep(3000);
            firstFloorHistogram.Invoke(
                new Action(() =>
                {
                    firstFloorHistogram.Size = new System.Drawing.Size(1000, 700);
                })
            );
            
            // Continuous Variable #1-2: Log of First Floor Square Feet
            var logFirstFloorHistogram = HistogramBox
            .Show(
                houseDF.DropSparseRows()["1stFlrSF"].Log().ValuesAll.ToArray(),
                title: "First Floor Square Feet - Log Transformed (Continuous)"
            )
            .SetNumberOfBins(20);

            System.Threading.Thread.Sleep(3000);
            logFirstFloorHistogram.Invoke(
                new Action(() =>
                {
                    logFirstFloorHistogram.Size = new System.Drawing.Size(1000, 700);
                })
            );

            // Continuous Variable #2-1: Size of garage in square feet
            var garageHistogram = HistogramBox
            .Show(
                houseDF.DropSparseRows()["GarageArea"].ValuesAll.ToArray(),
                title: "Size of garage in square feet (Continuous)"
            )
            .SetNumberOfBins(20);

            System.Threading.Thread.Sleep(3000);
            garageHistogram.Invoke(
                new Action(() =>
                {
                    garageHistogram.Size = new System.Drawing.Size(1000, 700);
                })
            );

            // Continuous Variable #2-2: Log of Value of miscellaneous feature
            var logGarageHistogram = HistogramBox
            .Show(
                houseDF.DropSparseRows()["GarageArea"].Log().ValuesAll.ToArray(),
                title: "Size of garage in square feet - Log Transformed (Continuous)"
            )
            .SetNumberOfBins(20);

            System.Threading.Thread.Sleep(3000);
            logGarageHistogram.Invoke(
                new Action(() =>
                {
                    logGarageHistogram.Size = new System.Drawing.Size(1000, 700);
                })
            );

            // Target Variable: Sale Price
            var salePriceHistogram = HistogramBox
            .Show(
                houseDF.DropSparseRows()["SalePrice"].ValuesAll.ToArray(),
                title: "Sale Price (Continuous)"
            )
            .SetNumberOfBins(20);

            System.Threading.Thread.Sleep(3000);
            salePriceHistogram.Invoke(
                new Action(() =>
                {
                    salePriceHistogram.Size = new System.Drawing.Size(1000, 700);
                })
            );

            // Target Variable: Sale Price - Log Transformed
            var logSalePriceHistogram = HistogramBox
            .Show(
                houseDF.DropSparseRows()["SalePrice"].Log().ValuesAll.ToArray(),
                title: "Sale Price - Log Transformed (Continuous)"
            )
            .SetNumberOfBins(20);

            System.Threading.Thread.Sleep(3000);
            logSalePriceHistogram.Invoke(
                new Action(() =>
                {
                    logSalePriceHistogram.Size = new System.Drawing.Size(1000, 700);
                })
            );


            Console.WriteLine("\nDONE!!!");
            Console.ReadKey();
        }
    }
}
