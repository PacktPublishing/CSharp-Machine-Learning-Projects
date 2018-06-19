using System;
using System.IO;
using Deedle;
using System.Linq;
using System.Text.RegularExpressions;

namespace DataProcessor
{
    class Program
    {
        private static string CleanTweet(string rawTweet)
        {
            string eyesPattern = @"[8:=;]";
            string nosePattern = @"['`\-]?";

            string tweet = rawTweet;
            // 1. Remove URL's
            string urlPattern = @"https?:\/\/\S+\b|www\.(\w+\.)+\S*";
            Regex rgx = new Regex(urlPattern);
            tweet = rgx.Replace(tweet, "");
            // 2. Remove Twitter ID's
            string userIDPattern = @"@\w+";
            rgx = new Regex(userIDPattern);
            tweet = rgx.Replace(tweet, "");
            // 3. Replace Smiley Faces
            string smileyFacePattern = String.Format(@"{0}{1}[)dD]+|[)dD]+{1}{0}", eyesPattern, nosePattern);
            tweet = Regex.Replace(tweet, smileyFacePattern, " emo_smiley ");
            // 4. Replace LOL Faces
            string lolFacePattern = String.Format(@"{0}{1}[pP]+", eyesPattern, nosePattern);
            tweet = Regex.Replace(tweet, lolFacePattern, " emo_lol ");
            // 5. Replace Sad Faces
            string sadFacePattern = String.Format(@"{0}{1}\(+|\)+{1}{0}", eyesPattern, nosePattern);
            tweet = Regex.Replace(tweet, sadFacePattern, " emo_sad ");
            // 6. Replace Neutral Faces
            string neutralFacePattern = String.Format(@"{0}{1}[\/|l*]", eyesPattern, nosePattern);
            tweet = Regex.Replace(tweet, neutralFacePattern, " emo_neutral ");
            // 7. Replace Heart
            string heartPattern = "<3";
            tweet = Regex.Replace(tweet, heartPattern, " emo_heart ");
            // 8. Replace Punctuation Repeat
            string repeatedPunctuationPattern = @"([!?.]){2,}";
            tweet = Regex.Replace(tweet, repeatedPunctuationPattern, " $1_repeat ");
            // 9. Replace Elongated Words (i.e. wayyyy -> way_emphasized)
            string elongatedWordsPattern = @"\b(\S*?)(.)\2{2,}\b";
            tweet = Regex.Replace(tweet, elongatedWordsPattern, " $1$2_emphasized ");
            // 10. Replace Numbers
            string numberPattern = @"[-+]?[.\d]*[\d]+[:,.\d]*";
            tweet = Regex.Replace(tweet, numberPattern, "");
            // 11. Replace Hashtag
            string hashtagPattern = @"#";
            tweet = Regex.Replace(tweet, hashtagPattern, "");

            return tweet;
        }

        private static string[] FormatTweets(Series<int, string> rows)
        {
            var cleanTweets = rows.GetAllValues().Select((x, i) =>
            {
                string tweet = x.Value;
                return CleanTweet(tweet);
            });

            return cleanTweets.ToArray();
        }

        static void Main(string[] args)
        {
            Console.SetWindowSize(250, 80);

            // Read in the sample set file
            // TODO: change the path to point to your data directory
            string dataDirPath = @"<path-to-data-dir>";

            // Load the data into a data frame
            string trainDataPath = Path.Combine(dataDirPath, "Tweets.csv");
            Console.WriteLine("Loading {0}", trainDataPath);
            var rawDF = Frame.ReadCsv(
                trainDataPath,
                hasHeaders: true,
                inferTypes: true
            );

            // Look at words used in tweets
            Console.WriteLine("Processing raw tweets...");

            // Clean the tweets
            string[] processedTweets = FormatTweets(rawDF.GetColumn<string>("text"));
            rawDF.AddColumn("tweet", processedTweets);
            rawDF.SaveCsv(Path.Combine(dataDirPath, "processed-training.csv"));

            Console.WriteLine("* Processed DF Shape ({0}, {1})", rawDF.RowCount, rawDF.ColumnCount);
            rawDF.GetRowsAt(new[] { 0, 1, 2, 3, 4 }).Print();

            Console.WriteLine("DONE!!");
            Console.ReadKey();
        }
    }
}
