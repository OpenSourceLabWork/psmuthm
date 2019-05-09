package org.uthm;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class Classifier {
	static SupportVectorMachine svmc;
	static NaiveBayesClassifier nbc;
	
	public static void main(String[] args) throws Exception{
		
		DataSource source = new DataSource("datasets/iris.arff");
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		svmc = new SupportVectorMachine();
		nbc = new NaiveBayesClassifier();
		
		
		String svmResult = svmc.run(dataset);
		System.out.println(svmResult);
		
		String nbcResult = nbc.run(dataset);
		System.out.println(nbcResult);
		
		
		/*
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(dataset);
		Evaluation eval = new Evaluation(dataset);
		eval.evaluateModel(nb, dataset);
		System.out.println(eval.toSummaryString());
		
		SMO svm = new SMO();
		svm.buildClassifier(dataset);
		Evaluation eval2 = new Evaluation(dataset);
		eval2.evaluateModel(svm, dataset);
		System.out.println(eval2.toSummaryString());
		
		LibSVM libsvm = new LibSVM();
		String[] options = new String[8];
		options[0] = "-S"; options[1] = "0";
		options[2] ="-K"; options[3] = "2";
		options[4] = "-G"; options[5] = "1.0";
		options[6] = "-C"; options[7] = "1.0";
	    libsvm.setOptions(options);
		libsvm.buildClassifier(dataset);
		Evaluation eval3 = new Evaluation(dataset);
		eval3.evaluateModel(libsvm, dataset);
		System.out.println(eval3.toSummaryString());
		*/
	}
}
