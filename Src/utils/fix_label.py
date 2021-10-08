import re

'''
This file is adapted from the repository https://github.com/jasonwu0731/trade-dst
'''

fin = open('utils/mapping.pair','r')
replacements = []
for line in fin.readlines():
	tok_from, tok_to = line.replace('\n', '').split('\t')
	replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))


def fix_general_label_error(labels, type, slots):
	if not type:
		label_dict = dict([ (l["slots"][0][0], l["slots"][0][1]) for l in labels])
	else:
		label_dict = labels

	GENERAL_TYPO = {
		# type
		"guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports", 
		"sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall", 
		"concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
		"colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
		# area
		"center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east", 
		"east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
		"city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north", 
		"centre of town":"centre", "cb30aq": "none",
		# price
		"mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", 
		# day
		"next friday":"friday", "monda": "monday", 
		# parking
		"free parking":"free",
		# internet
		"free internet":"yes",
		# star
		"4 star":"4", "4 stars":"4", "0 star rarting":"none",
		# others 
		"y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
		'':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",  

		# new found in test set, andy
		"the gallery at 12": "gallery at 12 a high street",
		"school": "old schools",
		"old school": "old schools", 
		"cafe uno": "caffe uno",
		"caffee uno": "caffe uno",
		"mi": "christ s college", 
		"christ college": "christ s college",
		"mic": "michaelhouse cafe",
		"1530": "15:30",
		"cambridge corn exchange": "the cambridge corn exchange",
		"corn cambridge exchange": "the cambridge corn exchange",
		"huntington marriott": "huntingdon marriott hotel", 
		"huntingdon hotel": "huntingdon marriott hotel",
		"huntingdon marriot hotel": "huntingdon marriott hotel",
		"ian hong house": "lan hong house",
		"lan hong": "lan hong house",
		"margherita": "la margherita",
		"india": 'indian',
		"alexander": "alexander bed and breakfast", 
		"alex": "alexander bed and breakfast",
		"the alex": "alexander bed and breakfast",
		"alexander bed & breakfast": "alexander bed and breakfast",
		"the alexander bed and breakfast": "alexander bed and breakfast",
		"charlie": "charlie chan",
		"finches": "finches bed and breakfast",
		"sleeperz hotel": "cityroomz",
		"4:45": "04:45",
		"pipasha": 'pipasha restaurant',
		"shanghi family restaurant": "shanghai family restaurant",
		"09,45": "09:45",
		"after 5:45 pm": "5:45",
		"wandlebury coutn": "a and b guest house",
		"scott polar": "scott polar museum",
		"cambrid": "cambride",
		"02:45 .": "02:45"
		}

	# remove wrong value label, andy
	WRONG_VALUE = ['cambridge be', 'dif', 'city stop rest', 'belf', 'hol', 'doubletree by hilton cambridge']
	remove_slots = []
	for slot, value in label_dict.items():
		if value in WRONG_VALUE:
			remove_slots.append(slot)
	for slot in remove_slots:
		del label_dict[slot]

	for slot in slots:
		if slot in label_dict.keys():
			# general typos
			if label_dict[slot] in GENERAL_TYPO.keys():
				label_dict[slot] = label_dict[slot].replace(label_dict[slot], GENERAL_TYPO[label_dict[slot]])
			
			# miss match slot and value 
			if  slot == "hotel-type" and label_dict[slot] in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
				slot == "hotel-internet" and label_dict[slot] == "4" or \
				slot == "hotel-pricerange" and label_dict[slot] == "2" or \
				slot == "attraction-type" and label_dict[slot] in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
				"area" in slot and label_dict[slot] in ["moderate"] or \
				"day" in slot and label_dict[slot] == "t":
				label_dict[slot] = "none"
			elif slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
				label_dict[slot] = "hotel"
			elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
				label_dict[slot] = "3"
			elif "area" in slot:
				if label_dict[slot] == "no": label_dict[slot] = "north"
				elif label_dict[slot] == "we": label_dict[slot] = "west"
				elif label_dict[slot] == "cent": label_dict[slot] = "centre"
			elif "day" in slot:
				if label_dict[slot] == "we": label_dict[slot] = "wednesday"
				elif label_dict[slot] == "no": label_dict[slot] = "none"
			elif "price" in slot and label_dict[slot] == "ch":
				label_dict[slot] = "cheap"
			elif "internet" in slot and label_dict[slot] == "free":
				label_dict[slot] = "yes"

			# some out-of-define classification slot values
			if  slot == "restaurant-area" and label_dict[slot] in ["stansted airport", "cambridge", "silver street"] or \
				slot == "attraction-area" and label_dict[slot] in ["norwich", "ely", "museum", "same area as hotel"]:
				label_dict[slot] = "none"

	return label_dict

def insertSpace(token, text):
	sidx = 0
	while True:
		sidx = text.find(token, sidx)
		if sidx == -1:
			break
		if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
				re.match('[0-9]', text[sidx + 1]):
			sidx += 1
			continue
		if text[sidx - 1] != ' ':
			text = text[:sidx] + ' ' + text[sidx:]
			sidx += 1
		if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
			text = text[:sidx + 1] + ' ' + text[sidx + 1:]
		sidx += 1
	return text

def dst_normalize(text, clean_value=True):
	# lower case every word
	text = text.lower()

	# replace white spaces in front and end
	text = re.sub(r'^\s*|\s*$', '', text)

	# hotel domain pfb30
	text = re.sub(r"b&b", "bed and breakfast", text)
	text = re.sub(r"b and b", "bed and breakfast", text)

	if clean_value:
		# normalize phone number
		ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
		if ms:
			sidx = 0
			for m in ms:
				sidx = text.find(m[0], sidx)
				if text[sidx - 1] == '(':
					sidx -= 1
				eidx = text.find(m[-1], sidx) + len(m[-1])
				text = text.replace(text[sidx:eidx], ''.join(m))

		# normalize postcode
		ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
						text)
		if ms:
			sidx = 0
			for m in ms:
				sidx = text.find(m, sidx)
				eidx = sidx + len(m)
				text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

	# weird unicode bug
	text = re.sub(u"(\u2018|\u2019)", "'", text)

	if clean_value:
		# replace time and and price
		text = re.sub(timepat, ' [value_time] ', text)
		text = re.sub(pricepat, ' [value_price] ', text)
		#text = re.sub(pricepat2, '[value_price]', text)

	# replace st.
	text = text.replace(';', ',')
	text = re.sub('$\/', '', text)
	text = text.replace('/', ' and ')

	# replace other special characters
	text = text.replace('-', ' ')
	text = re.sub('[\"\<>@\(\)]', '', text) # remove

	# insert white space before and after tokens:
	for token in ['?', '.', ',', '!']:
		text = insertSpace(token, text)

	# insert white space for 's
	text = insertSpace('\'s', text)

	# replace it's, does't, you'd ... etc
	text = re.sub('^\'', '', text)
	text = re.sub('\'$', '', text)
	text = re.sub('\'\s', ' ', text)
	text = re.sub('\s\'', ' ', text)
	for fromx, tox in replacements:
		text = ' ' + text + ' '
		text = text.replace(fromx, tox)[1:-1]

	# remove multiple spaces
	text = re.sub(' +', ' ', text)

	# concatenate numbers
	tmp = text
	tokens = text.split()
	i = 1
	while i < len(tokens):
		if re.match(u'^\d+$', tokens[i]) and \
				re.match(u'\d+$', tokens[i - 1]):
			tokens[i - 1] += tokens[i]
			del tokens[i]
		else:
			i += 1
	text = ' '.join(tokens)

	return text
