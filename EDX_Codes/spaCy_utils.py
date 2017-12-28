import spacy
from spacy.symbols import advmod, VERB
nlp = spacy.load('en')

# doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
# doc = nlp(u'What was the first woman inhumanely killed and brutally assaulted in the horrific Vietnam War?')

doc = nlp(u'Credit and mortgage account holders must submit their requests')
span = doc[doc[4].left_edge.i : doc[4].right_edge.i+1]
span.merge()
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)

# # for x in doc.noun_chunks:
# # 	print(x)

# for token in doc:
# 	print(token.text, token.pos_, token.dep_)
# #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
# #           token.shape_, token.is_alpha, token.is_stop)
		
# for token in doc:
#     if token.pos == VERB:
#     	# print([x for x in token.children])
#     	for child in token.children:
#             if child.dep_ == advmod:
#                 verbs.append(child)
#                 break

# print(verbs)
# def get_head_chunk():
# 	flag = 0
# 	for token in doc:
# 		if flag == 1 and flag < 3:
# 			if token.pos_ == VERB:
# 				token.

# 		if(token.tag_.startswith('W')):
# 			# print(token.text, token.pos_, token.tag_)
# 			flag = 1


# get_head_chunk()