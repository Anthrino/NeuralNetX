import numpy as np

glove_wordmap = {}
v, m, RS = None, None, None

def get_glove(vector_dimension = 50):
	global glove_wordmap
	print('>>> Loading GloVe')
	glove_file_path = path.join('../../EruditeX/EruditeX/data/glove/glove.6B.{}d.txt'.format(str(vector_dimension)))
	with open(glove_file_path, 'r', encoding = 'UTF-8') as f:
		for line in f:
			word, vector = tuple(line.split(' ', 1))
			glove_wordmap[word] = np.fromstring(vector, sep=' ')
	global v
	global m
	global RS
	v, m, RS = set_variables_for_unknown()
	return glove_wordmap


def set_variables_for_unknown():
	wvecs = []
	for item in glove_wordmap.items():
		wvecs.append(item[1])
	s = np.vstack(wvecs)

	v = np.var(s, 0)
	m = np.mean(s, 0)
	RS = np.random.RandomState()
	return v, m, RS


def fill_unknown(unk):
	global glove_wordmap
	global n
	global v
	global RS
	glove_wordmap[unk] = RS.multivariate_normal(m, np.diag(v))
	return glove_wordmap[unk]

