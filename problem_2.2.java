package org.example;

import java.io.*;
import java.util.*;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import opennlp.tools.stemmer.PorterStemmer;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
public class TermFrequency {
    // To store the Document frequency values
    // From the file in cache for fast look up during
    // SCORE computation.(global for access everywhere
    // within the TermFrequency class.
    public static HashMap<String, Integer> docHash = new HashMap<>();

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable>
    {
        private Text word = new Text();
        private IntWritable value = new IntWritable(1);
        // Setup method is used to retrieve and use cached files.

        @Override
        public void setup(Context context) throws IOException, InterruptedException
        {
            // Retrieving all the cached files from the distributed cache.

            URI[] cacheFiles = DistributedCache.getCacheFiles(context.getConfiguration());
            if(cacheFiles.length > 0){
                try {
                    FileSystem fileSystem = FileSystem.get(context.getConfiguration());
                    Path filePath = new Path(cacheFiles[0].toString());
                    BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fileSystem.open(filePath)));
                    String line = "";
                    // while the bufferedReader instance is able to
                    // retrienve lines from the cached file
                    // split the input and store the Key Value pair of
                    // Document Frequency into the HashMap created above.
                    while ((line = bufferedReader.readLine()) != null) {
                        String[] split = line.split("\t");
                        docHash.put(split[0], Integer.parseInt(split[1]));
                    }
                    bufferedReader.close();

                }


                // catching any errors while reading the file.
                catch (IOException e){
                    System.out.println("Error occured while reading file");
                    System.exit(1);
                }
            }
            else{
                System.out.println("NO FILES CACHED!!! PLEASE GIVE CORRECT ARGUMENTS");
                System.exit(1);
            }
        }

        @Override
        public void map(Object Key, Text Value, Context context) throws IOException, InterruptedException {
            String line = Value.toString();
            // Split the line into constituent tokens based on separator.
            String[] tokens = line.split("[^\\w']+");
            // Initialising stemmer to perform stemming on the words
            // as per requirement,
            PorterStemmer stemmer = new PorterStemmer();
            String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();
            for(String token : tokens){
//                if(docHash.containsKey(token)==false){
//                    continue;
//                }
                String stemmed = stemmer.stem(token).toString();
                if(docHash.containsKey(stemmed)==false){
                    continue;
                }
                String finalKey = stemmed + " "+ fileName;
                word.set(finalKey);
                context.write(word, value);
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, FloatWritable> {
        // Variable to store the computed SCORE.
        private FloatWritable result = new FloatWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum =0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            String[] key_word = key.toString().split(" ");
            // Looking up the document frequency value for the word in hashmap.
            // int dfValue = docHash.get(key_word[0]);
            int dfValue = docHash.getOrDefault(key_word[0], 0);
            // Score computation based on the formula for tfidf given in
            // the requirement document.
            float tfidfScore = sum * (float)Math.log(1000/(dfValue+1));
            // Setting tab spaces between as required for tab separated values file.
            String final_key = key_word[1]+ "\t"+ key_word[0];
            result.set(tfidfScore);
            key.set(final_key);
            context.write(key, result);
        }
    }
    public static void main(String[] args) throws Exception {
        // Setting up of the job to execute the program.
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "TermFrequency");
        // setting the mapper class for job
        job.setJarByClass(TermFrequency.class);
        job.setMapperClass(TokenizerMapper.class);
        // setting the reducer class for job
        job.setReducerClass(IntSumReducer.class);
        // setting the output file datatypes for Key, Value pair
        // Format: Text, IntWritable
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        // The arguments given while execution is used
        // to set the cache file and the input and output paths.
        // Format: hadoop jar <jar path> className input path cachefilepath output path
        // Adds the document frequency file into the cache
        // this helps us perform fast look up.
        DistributedCache.addCacheFile(new Path(args[1]).toUri(),job.getConfiguration());

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[2]));
        // The program will run till the job is completed.
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
