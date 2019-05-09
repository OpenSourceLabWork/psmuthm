package org.uthm;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class NaiveBayesClassifier {
	
	public String run(Instances dataset){
		
		NaiveBayes nb = new NaiveBayes();
		Evaluation eval = null;
		
		try {
			nb.buildClassifier(dataset);
			eval = new Evaluation(dataset);
			eval.evaluateModel(nb, dataset);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return eval.toSummaryString();
	}

}
