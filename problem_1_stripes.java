package org.example;

import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.tokenize.SimpleTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;

public class PosTaggingStripes extends Configured implements Tool {
    public static void main(String[] args) throws Exception {
        Configuration configuration = new Configuration();
        String[] otherArgs = new GenericOptionsParser(configuration, args).getRemainingArgs();
        int res = ToolRunner.run(new Configuration(), new PosTaggingStripes(), otherArgs);
        System.exit(res);
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = Job.getInstance(getConf(), "POS Tagging Stripes");
        job.setJarByClass(PosTaggingStripes.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setReducerClass(StripeReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(MapWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static class TokenizerMapper extends Mapper<LongWritable, Text, Text, MapWritable> {
        private SimpleTokenizer tokenizer;
        private POSTaggerME tagger;

        @Override
        protected void setup(Context context) {
            try {
                InputStream modelIn = getClass().getResourceAsStream("/model.bin");

                if (modelIn == null) {
                    throw new FileNotFoundException("Model file not found.");
                }

                // Load the POS tagging model
                POSModel model = new POSModel(modelIn);

                tokenizer = SimpleTokenizer.INSTANCE;
                tagger = new POSTaggerME(model);

            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = tokenizer.tokenize(value.toString());
            String[] tags = tagger.tag(tokens);

            MapWritable stripe = new MapWritable();

            for (String tag : tags) {
                Text t = new Text(tag);

                if (stripe.containsKey(t)) {
                    IntWritable v = (IntWritable) stripe.get(t);
                    v.set(v.get() + 1);
                } else {
                    stripe.put(t, new IntWritable(1));
                }
            }

            context.write(new Text("stripe"), stripe);

        }
    }

    public static class StripeReducer extends Reducer<Text, MapWritable, Text, MapWritable> {

        @Override
        protected void reduce(Text key, Iterable<MapWritable> values, Context context)
                throws IOException, InterruptedException {
            MapWritable result = new MapWritable();

            for (MapWritable stripe : values) {
                for (MapWritable.Entry<Writable, Writable> entry : stripe.entrySet()) {
                    Text tag = (Text) entry.getKey();
                    IntWritable v = (IntWritable) entry.getValue();

                    if (result.containsKey(tag)) {
                        IntWritable count = (IntWritable) result.get(tag);
                        count.set(count.get() + v.get());
                    } else {
                        result.put(tag, v);
                    }

                }
            }

            context.write(key, result);
        }
    }
}
