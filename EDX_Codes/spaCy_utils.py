import spacy
import textacy
from spacy import displacy
from nltk import Tree
nlp = spacy.load('en_core_web_sm')

# sent = 'Who was the first woman inhumanely killed and brutally assaulted in the horrific Vietnam War?'
sent = 'Apple is looking at buying U.K. startup for $1 billion'
# sent = 'Credit and mortgage account holders must submit their requests'

# def get_dtree(sent):
	# doc = nlp(sent)
	# # displacy.serve(doc, style='dep')
	# sents = [sent for sent in doc.sents]
	# sent = sents[0]
	# print(sent.root)
	# for token in doc:
	# 	print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
	# 	print(token.subtree)

class dt_node:

	def __init__(self, node, children):
		self.text = node.text
		self.pos_tag = node.pos_
		self.dep_tag = node.dep_
		self.head = node.head.text
		# self.word_vector = get_vector(node.text, load_glove())
		self.hid_state = None
		self.children = children

	def get_text(self):
		return self.text

	def get_children(self):
		return self.children

	def has_children(self):
		return not(self.children == [])

	def count_nodes(self):

		count = 0
		if self.has_children():
			for cnode in self.children:
				count += cnode.count_nodes()
		return 1 + count

	def count_non_leaf(self):
		if self.is_leaf():
			return 0
		left_count = 0
		right_count = 0
		if self.has_left_child():
			left_count = self.get_left_child().count_non_leaf()
		if self.has_right_child():
			right_count = self.get_right_child().count_non_leaf()
		return 1 + left_count + right_count

	def inorder_print(self):
		if self.has_left_child():
			self.get_left_child().inorder_print()
		print(self.get_value())
		if self.has_right_child():
			self.get_right_child().inorder_print()
		return

	def preorder_print(self):
		print(self.get_value())
		if self.has_left_child():
			self.get_left_child().preorder_print()
		if self.has_right_child():
			self.get_right_child().preorder_print()
		return

	def postorder(self):

		po_list = []			

		if self.has_children():
			for cnode in self.children:
			 	for c in cnode.postorder():
			 		po_list.append(c)
	
		po_list.append(self)


		return po_list

	def get_tree_traversal(self, mode):
		
		node_list = []
		po_list = self.postorder()
		
		for node in po_list:

			if mode == 'head-index':
				count = 0
				for n in po_list:
					if n.text == node.head:
						node_list.append(count)
						break
					else:
						count+=1
			
			elif mode == 'text':
				node_list.append(node.text)
			
			# else if mode == 'vector':
		return node_list

	def get_path(self, value, path):
		if self.get_value() == value:
			path.append(self.get_value())
			return True
		if self.has_left_child() and self.get_left_child().get_path(value, path):
			path.insert(0, self.get_value())
			return True
		if self.has_right_child() and self.get_right_child().get_path(value, path):
			path.insert(0, self.get_value())
			return True
		return False

	def get_path_from_root(self, value):
		path = []
		if self.get_path(value, path):
			return path
		return None
 	
def get_dtree(sentence):
	doc = nlp(sentence)
	sents = [sent for sent in doc.sents]
	[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
	sent = sents[0]
	return get_tree_node(sent.root)

def get_tree_node(node):
	if node.children:
		return dt_node(node, [get_tree_node(child) for child in node.children])
	else:
		return node

def get_verb_chunks(sent):
	verb_chunks = []
	pattern = r'<VERB>?<ADV>*<VERB>+'
	doct = textacy.Doc(sent, lang='en_core_web_sm')
	lists = textacy.extract.pos_regex_matches(doct, pattern)
	for list in lists:
		verb_chunks.append(list.text)
	# print(list.text)

	print(verb_chunks)

	return verb_chunks

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


root = get_dtree(sent)
print(root.get_tree_traversal('head-index'))
print(root.get_tree_traversal('text'))
print(root.get_children(), root.count_nodes())

# def get_head_chunk():

# flag = 0

# for token in doc:
# 	if flag == 1 and flag < 3:
# 		if token.pos_ == VERB and token.text in verb_chunks:
# 			print(token.text)
# 			flag += 1

# 	if(token.tag_.startswith('W')):
# 		# print(token.text, token.pos_, token.tag_)
# 		flag = 1
# for token in doc:
#     print(token.text, token.pos_, token.dep_, token.head.text)

# for x in doc.noun_chunks:
# 	print(x)

# for token in doc:
#     if token.pos == VERB:
#     	# print([x for x in token.children])
#     	for child in token.children:
#             if child.dep_ == advmod:
#                 verbs.append(child)
#                 break
# print(verbs)
