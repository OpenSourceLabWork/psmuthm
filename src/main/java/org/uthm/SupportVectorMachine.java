package org.uthm;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

public class SupportVectorMachine {
	
	public String run(Instances train, Instances test){
		SMO svm = new SMO();
		Evaluation eval2 = null;
		try {
			train.setClassIndex(train.numAttributes() - 1);
			test.setClassIndex(train.numAttributes() - 1);
			
			svm.buildClassifier(train);
			eval2 = new Evaluation(train);
			eval2.evaluateModel(svm, train);
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
				
		return eval2.toSummaryString();
	}

}
