package org.uthm;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class DecisionTreeClassifier {
	
	public String run(Instances train, Instances test){
		
		J48 tree = new J48();
		Evaluation eval = null;
//		System.out.println("Reach run method");
			
			try {
				train.setClassIndex(train.numAttributes() - 1);
				tree.buildClassifier(train);
				eval = new Evaluation(train);
				eval.evaluateModel(tree, train);
				
				
//				System.out.println("Reach try catch");
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			return eval.toSummaryString();
		}

//	public String run2(Instances dataset) {
//
//		Id3 tree = new Id3();
//		Evaluation evaluation = null;
//
//		try {
//			// initialize the info gain extractor
//			InfoGainAttributeEval eval = new InfoGainAttributeEval();
//			Ranker search = new Ranker();
//
//			AttributeSelection attSelect = new AttributeSelection();
//			attSelect.setEvaluator(eval);
//			attSelect.setSearch(search);
//			attSelect.SelectAttributes(dataset);
//
//			// let's show the information gain value for each attribute
//			System.out.println(attSelect.toResultsString());
//
//			// now we build and show the decision tree
//
//			tree.buildClassifier(dataset);
//
//			evaluation = new Evaluation(dataset);
//			evaluation.evaluateModel(tree, dataset);
//		} catch (Exception ex) {
//			ex.printStackTrace();
//		}
//
//		return evaluation.toSummaryString();
//	}

}
