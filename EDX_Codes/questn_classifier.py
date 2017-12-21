'''

	Hierarchical Question Classifier 

	Two layers : Coarse and Fine classifiers

	C0 (initial set of question type classes) -> // Coarse Classifier (Probability/Density of each class) // -> C1 (Top 5 dense / over threshold classes from C0)

	C1 -> C2 (expansion of coarse classes into subclasses) -> // Fine Classifier (Probability/Density of each subclass) // C3 (Final subclass labels)

 '''

 