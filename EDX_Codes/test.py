import csv

questions = []
answers = []
with open(".\Dataset\WikiQACorpus\WikiQA-dev.tsv", encoding="utf8") as data_file:
	source = list(csv.reader(data_file, delimiter="\t", quotechar='"'))
	q_index = 'Q-1'
	ans_sents = []

	for row in source[1:]:

		if q_index != row[0]:
			answers.append(ans_sents)
			ans_sents = []
			questions.append(row[1])
			q_index = row[0]

		ans_sents.append({row[5]: row[6]})

	answers.append(ans_sents)
	answers = answers[1:]

	for i in range(len(questions)):
		print("Question:", questions[i])
		print("Answers:", answers[i])
