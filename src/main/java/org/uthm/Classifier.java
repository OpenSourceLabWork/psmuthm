package org.uthm;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class Classifier {
	static SupportVectorMachine svmc;
	static NaiveBayesClassifier nbc;
	static DecisionTreeClassifier dtc;
	
	public static void main(String[] args) throws Exception{
		
		// CSVLoader loader = new CSVLoader();
		// loader.setSource(App.class.getResourceAsStream("weather.csv"));
		// Instances data = loader.getDataSet();

		
		
		DataSource source_train = new DataSource("datasets/iris-train.arff");
		Instances dataset_train = source_train.getDataSet();
//		dataset_train.setClassIndex(dataset_train.numAttributes()-1);
		
		DataSource source_test = new DataSource("datasets/iris-unknown.arff");
		Instances dataset_test = source_test.getDataSet();
		
		
		
		svmc = new SupportVectorMachine();
		nbc = new NaiveBayesClassifier();
		dtc = new DecisionTreeClassifier();
		
		System.out.println("_______________________________________________");
		System.out.println("Decision Tree classification results");
		String dtResult = dtc.run(dataset_train,dataset_test);
		System.out.println(dtResult);
		System.out.println("_______________________________________________");
		
		System.out.println("SVM classification results");
		String svmResult = svmc.run(dataset_train,dataset_test);
		System.out.println(svmResult);
		
		System.out.println("_______________________________________________");
		System.out.println("NaiveBayes classification results");
		String nbcResult = nbc.run(dataset_train,dataset_test);
		System.out.println(nbcResult);
		System.out.println("_______________________________________________");
	
		
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
