
def get_labels(filepath):

	questions = []
	coarse_class = []
	fine_class = []
	with open(filepath) as source:
		source_text = source.read()

	for line in source_text.split('\n'):
		
		labels, question = line.split(' ', 1)
		c_class, f_class = labels.split(':')			
		# print(c_class, f_class, question)
		questions.append(question)
		coarse_class.append(c_class)
		fine_class.append(f_class)

	return questions, coarse_class, fine_class	

# print(get_labels(".\\Dataset\\trec_train_1000.txt"))