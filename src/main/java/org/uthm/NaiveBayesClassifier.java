package org.uthm;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class NaiveBayesClassifier {
	
	public String run(Instances train, Instances test){
		
		NaiveBayes nb = new NaiveBayes();
		Evaluation eval = null;
		
		try {
			train.setClassIndex(train.numAttributes() - 1);
			test.setClassIndex(train.numAttributes() - 1);
			
			nb.buildClassifier(train);
			eval = new Evaluation(train);
			eval.evaluateModel(nb, train);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return eval.toSummaryString();
	}

}
