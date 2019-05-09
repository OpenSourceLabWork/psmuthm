package org.uthm;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

public class SupportVectorMachine {
	
	public String run(Instances dataset){
		SMO svm = new SMO();
		Evaluation eval2 = null;
		try {
			svm.buildClassifier(dataset);
			eval2 = new Evaluation(dataset);
			eval2.evaluateModel(svm, dataset);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
				
		return eval2.toSummaryString();
	}

}
