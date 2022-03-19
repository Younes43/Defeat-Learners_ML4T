""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			     			  	 
All Rights Reserved  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			     			  	 
or edited.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			     			  	 
GT honor code violation.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import numpy as np  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
class DTLearner(object):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    This is a Decision tree Learner. It is implemented correctly.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  			  		 			     			  	 
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  	  			  		 			     			  	 
    :type verbose: bool  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    def __init__(self, leaf_size=1,verbose=False):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Constructor method  		  	   		  	  			  		 			     			  	 
        """  		
        self.leaf_size=leaf_size  
        self.verbose = verbose	 
        self.verboseprint = print if verbose else lambda *a, **k: None  		  	  			  		 			     			  	 
        #pass  # move along, these aren't the drones you're looking for  
        # 		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    def author(self):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        :return: The GT username of the student  		  	   		  	  			  		 			     			  	 
        :rtype: str  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        return "ybouzekraoui3"  # replace tb34 with your Georgia Tech username  		  	   		  	  			  		 			     			  	 

    def feature_to_split_on(self, data_x,data_y):
        l=[abs(np.corrcoef(data_y,data_x[:,i])[0,1]) for i in range(data_x.shape[1])]
        return l.index(max(l))
    
    def build_tree(self,data_x,data_y):
        if data_x.shape[0]<=self.leaf_size : 
            return np.array([['Leaf',data_y.mean(),np.nan,np.nan]])
        if (data_y == data_y[0]).sum()==len(data_y) :
            return np.array([['Leaf',data_y.mean(),np.nan,np.nan]])
        else :  
            i = self.feature_to_split_on(data_x,data_y) 
            SplitVal = np.median(data_x[:,i])
            if SplitVal == max(data_x[:, i]):
                return np.array([['Leaf', np.mean(data_y), np.nan, np.nan]])
            lefttree = self.build_tree(data_x[data_x[:,i]<=SplitVal],data_y[data_x[:,i]<=SplitVal])
            righttree = self.build_tree(data_x[data_x[:,i]>SplitVal],data_y[data_x[:,i]>SplitVal])
            root = np.array([i ,SplitVal,1,lefttree.shape[0]+1])
            return np.vstack((root,lefttree,righttree))



    def add_evidence(self, data_x, data_y):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Add training data to learner  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        :param data_x: A set of feature values used to train the learner  		  	   		  	  			  		 			     			  	 
        :type data_x: numpy.ndarray  		  	   		  	  			  		 			     			  	 
        :param data_y: The value we are attempting to predict given the X data  		  	   		  	  			  		 			     			  	 
        :type data_y: numpy.ndarray  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        self.tree = self.build_tree(data_x,data_y)  		 			     			  	 
  		  	   		  	  			  		 			     			  	 	  			  		 			     			  	 
      	   		  	  			  		 			     			  	 
    def query(self, points):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Estimate a set of test points given the model we built.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  	  			  		 			     			  	 
        :type points: numpy.ndarray  		  	   		  	  			  		 			     			  	 
        :return: The predicted result of the input data according to the trained model  		  	   		  	  			  		 			     			  	 
        :rtype: numpy.ndarray  		  	   		  	  			  		 			     			  	 
        """  	
        y_predicted = []
        decision_tree = self.tree
        for row in range(points.shape[0]):
            node = 0
            while decision_tree[int(node), 0] != 'Leaf':
                i = decision_tree[int(node), 0]
                SplitVal = float(decision_tree[int(node), 1])
                if points[row,int(float(i))] <= SplitVal:
                    node += float(decision_tree[int(node), 2])
                else: 
                    node +=float(decision_tree[int(node), 3])
            y_predicted=np.append(y_predicted,float(decision_tree[int(node), 1]))
        return y_predicted			  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    print("the secret clue is 'zzyzx'")  		  	   		  	  			  		 			     			  	 
