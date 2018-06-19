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
        private static Frame<int, string> CreateCategories(Series<int, string> rows, string originalColName)
        {

            var categoriesByRows = rows.GetAllValues().Select((x, i) =>
            {
                // Encode the categories appeared in each row with 1
                var sb = new SeriesBuilder<string, int>();
                sb.Add(String.Format("{0}_{1}", originalColName, x.Value), 1);

                return KeyValue.Create(i, sb.Series);
            });

            // Create a data frame from the rows we just created
            // And encode missing values with 0
            var categoriesDF = Frame.FromRows(categoriesByRows).FillMissing(0);

            return categoriesDF;
        }

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

            string[] categoricalVars = new string[]
            {
                "Alley", "BldgType", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                "BsmtQual", "CentralAir", "Condition1", "Condition2", "Electrical", "ExterCond",
                "Exterior1st", "Exterior2nd", "ExterQual", "Fence", "FireplaceQu", "Foundation",
                "Functional", "GarageCond", "GarageFinish", "GarageQual", "GarageType", "Heating",
                "HeatingQC", "HouseStyle", "KitchenQual", "LandContour", "LandSlope", "LotConfig",
                "LotShape", "MasVnrType", "MiscFeature", "MSSubClass", "MSZoning", "Neighborhood",
                "PavedDrive", "PoolQC", "RoofMatl", "RoofStyle", "SaleCondition", "SaleType", "Street", "Utilities"
            };

            var featuresDF = Frame.CreateEmpty<int, string>();

            foreach(string col in houseDF.ColumnKeys)
            {
                if (categoricalVars.Contains(col))
                {
                    var categoryDF = CreateCategories(houseDF.GetColumn<string>(col), col);

                    foreach (string newCol in categoryDF.ColumnKeys)
                    {
                        featuresDF.AddColumn(newCol, categoryDF.GetColumn<int>(newCol));
                    }
                }
                else if (col.Equals("SalePrice"))
                {
                    featuresDF.AddColumn(col, houseDF[col]);
                    featuresDF.AddColumn("Log"+col, houseDF[col].Log());
                }
                else
                {
                    featuresDF.AddColumn(col, houseDF[col].Select((x, i) => x.Value.Equals("NA")? 0.0: (double) x.Value));
                }
            }

            string outputPath = Path.Combine(dataDirPath, "features.csv");
            Console.WriteLine("Writing features DF to {0}", outputPath);
            featuresDF.SaveCsv(outputPath);

            Console.WriteLine("\nDONE!!!");
            Console.ReadKey();
        }
    }
}
