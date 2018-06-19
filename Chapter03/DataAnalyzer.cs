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
        private static Series<string, double> ColumnWiseSum(Frame<int, string> frame, string exclude)
        {
            var sb = new SeriesBuilder<string, double>();
            foreach(string colname in frame.ColumnKeys)
            {
                double frequency = frame[colname].Sum();
                if (!colname.Equals(exclude))
                {
                    sb.Add(colname, frequency);
                }
            }

            return sb.ToSeries();
        }

        static void Main(string[] args)
        {
            Console.SetWindowSize(125, 50);
            // Read in the file we created in the Data Preparation (TwitterTokenizer) step
            // TODO: change the path to point to your data directory
            string dataDirPath = @"<path-to-data-dir>";

            // Load the twitter-lemma data into a data frame
            var tweetLemmaDF = Frame.ReadCsv(
                Path.Combine(dataDirPath, "tweet-lemma.csv"),
                hasHeaders: true,
                inferTypes: true
            );
            Console.WriteLine("* DF shape: ({0}, {1})", tweetLemmaDF.RowCount, tweetLemmaDF.ColumnCount);

            var sampleSetDistribution = tweetLemmaDF.GetColumn<string>(
                "tweet_polarity"
            ).GroupBy<string>(x => x.Value).Select(x => x.Value.KeyCount);
            int[] sampleSizes = sampleSetDistribution.Values.ToArray();
            int neutralSampleSize = sampleSizes[0];
            int positiveSampleSize = sampleSizes[1];
            int negativeSampleSize = sampleSizes[2];

            Console.WriteLine("* sentiment distribution - neutral: {0}, positive: {1}, negative: {2}", neutralSampleSize, positiveSampleSize, negativeSampleSize);

            var neutralTermFrequencies = ColumnWiseSum(
                tweetLemmaDF.Where(
                    x => x.Value.GetAs<int>("tweet_polarity") == 0
                ),
                "tweet_polarity"
            ).Sort().Reversed;

            var positiveTermFrequencies = ColumnWiseSum(
                tweetLemmaDF.Where(
                    x => x.Value.GetAs<int>("tweet_polarity") == 1
                ),
                "tweet_polarity"
            ).Sort().Reversed;

            var negativeTermFrequencies = ColumnWiseSum(
                tweetLemmaDF.Where(
                    x => x.Value.GetAs<int>("tweet_polarity") == 2
                ),
                "tweet_polarity"
            ).Sort().Reversed;

            // Look at Top 10 terms that appear in Neutral vs. Positive vs. Negative tweets
            var topN = 7;

            var neutralTermProportions = neutralTermFrequencies / neutralSampleSize;
            var positiveTermProportions = positiveTermFrequencies / positiveSampleSize;
            var negativeTermProportions = negativeTermFrequencies / negativeSampleSize;

            var topNeutralTerms = neutralTermProportions.Keys.Take(topN);
            var topNeutralTermsProportions = neutralTermProportions.Values.Take(topN);

            var topPositiveTerms = positiveTermProportions.Keys.Take(topN);
            var topPositiveTermsProportions = positiveTermProportions.Values.Take(topN);

            var topNegativeTerms = negativeTermProportions.Keys.Take(topN);
            var topNegativeTermsProportions = negativeTermProportions.Values.Take(topN);

            System.IO.File.WriteAllLines(
                dataDirPath + "\\neutral-frequencies.csv",
                neutralTermFrequencies.Keys.Zip(
                    neutralTermFrequencies.Values, (a, b) => string.Format("{0},{1}", a, b)
                )
            );
            System.IO.File.WriteAllLines(
                dataDirPath + "\\positive-frequencies.csv",
                positiveTermFrequencies.Keys.Zip(
                    positiveTermFrequencies.Values, (a, b) => string.Format("{0},{1}", a, b)
                )
            );
            System.IO.File.WriteAllLines(
                dataDirPath + "\\negative-frequencies.csv",
                negativeTermFrequencies.Keys.Zip(
                    negativeTermFrequencies.Values, (a, b) => string.Format("{0},{1}", a, b)
                )
            );

            var topNeutralBarChart = DataBarBox.Show(
                topNeutralTerms.ToArray(),
                new double[][] {
                    topNeutralTermsProportions.ToArray(),
                    negativeTermProportions.GetItems(topNeutralTerms).Values.ToArray(),
                    positiveTermProportions.GetItems(topNeutralTerms).Values.ToArray()
                }
            );
            topNeutralBarChart.SetTitle(
                String.Format(
                    "Top {0} Terms in Neutral Tweets (blue: neutral, red: negative, green: positive)",
                    topN
                )
             );
            System.Threading.Thread.Sleep(3000);
            topNeutralBarChart.Invoke(
                new Action(() =>
                {
                    topNeutralBarChart.Size = new System.Drawing.Size(5000, 1500);
                })
            );

            var topPositiveBarChart = DataBarBox.Show(
                topPositiveTerms.ToArray(),
                new double[][] {
                    neutralTermProportions.GetItems(topPositiveTerms).Values.ToArray(),
                    negativeTermProportions.GetItems(topPositiveTerms).Values.ToArray(),
                    topPositiveTermsProportions.ToArray()
                }
            );
            topPositiveBarChart.SetTitle(
                String.Format(
                    "Top {0} Terms in Positive Tweets (blue: neutral, red: negative, green: positive)",
                    topN
                )
             );
            System.Threading.Thread.Sleep(3000);
            topPositiveBarChart.Invoke(
                new Action(() =>
                {
                    topPositiveBarChart.Size = new System.Drawing.Size(5000, 1500);
                })
            );

            var topNegattiveBarChart = DataBarBox.Show(
                topNegativeTerms.ToArray(),
                new double[][] {
                    neutralTermProportions.GetItems(topNegativeTerms).Values.ToArray(),
                    topNegativeTermsProportions.ToArray(),
                    positiveTermProportions.GetItems(topNegativeTerms).Values.ToArray()
                }
            );
            topNegattiveBarChart.SetTitle(
                String.Format(
                    "Top {0} Terms in Negative Tweets (blue: neutral, red: negative, green: positive)",
                    topN
                )
             );
            System.Threading.Thread.Sleep(3000);
            topNegattiveBarChart.Invoke(
                new Action(() =>
                {
                    topNegattiveBarChart.Size = new System.Drawing.Size(5000, 1500);
                })
            );

            Console.WriteLine("Done");
            Console.ReadKey();
        }
    }
}
