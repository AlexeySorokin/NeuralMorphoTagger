import sys
from collections import defaultdict


WORD_COLUMN, POS_COLUMN, TAG_COLUMN = 1, 3, 5

POS_MAPPING = {".": "<SENT>", "?": "<QUESTION>", "!":"<EXCLAM>",
               ",": "<COMMA>", "-": "<HYPHEN>", "--": "<DASH>",
               ":": "COLON", ";": "SEMICOLON", "\"": "<QUOTE>"}

def process_word(word, to_lower=False, append_case=None):
    if all(x.isupper() for x in word):
        uppercase = "<ALL_UPPER>"
    elif word[0].isupper():
        uppercase = "<FIRST_UPPER>"
    else:
        uppercase = None
    if to_lower:
        word = word.lower()
    if word.isdigit():
        answer = ["<DIGIT>"]
    elif word.startswith("http://") or word.startswith("www."):
        answer = ["<HTTP>"]
    else:
        answer = list(word)
    if to_lower and uppercase is not None:
        if append_case == "first":
            answer = [uppercase] + answer
        elif append_case == "last":
            answer = answer + [uppercase]
    return tuple(answer)


def extract_frequent_words(infiles, to_lower=False, append_case="first", threshold=20,
                           relative_threshold=0.001, max_frequent_words=100):
    counts = defaultdict(int)
    for infile in infiles:
        with open(infile, "r", encoding="utf8") as fin:
            for line in fin:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                splitted = line.split("\t")
                index = splitted[0]
                if not index.isdigit():
                    continue
                word, pos, tag = splitted[WORD_COLUMN], splitted[POS_COLUMN], splitted[TAG_COLUMN]
                word = process_word(word, to_lower=to_lower, append_case=append_case)
                if pos not in ["SENT", "PUNCT"] and word != tuple("<DIGIT>"):
                    tag = "{},{}".format(pos, tag) if tag != "_" else pos
                    counts[(word, tag)] += 1
    total_count = sum(counts.values())
    threshold = max(relative_threshold * total_count, threshold)
    counts = [elem for elem in counts.items() if elem[1] >= threshold]
    counts = sorted(counts)[:max_frequent_words]
    frequent_pairs = set(elem[0] for elem in counts)
    return frequent_pairs


def read_tags_infile(infile, read_words=False, to_lower=False,
                     append_case="first", wrap=False,
                     attach_tokens=False, max_sents=-1):
    answer, curr_sent, curr_word_sent = [], [], []
    with open(infile, "r", encoding="utf8") as fin:
        print(infile)
        has_errors = False
        for line in fin:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line == "":
                if len(curr_sent) > 0:
                    answer.append((curr_word_sent, curr_sent))
                curr_sent, curr_word_sent = [], []
                if len(answer) == max_sents:
                    break
                continue
            splitted = line.split("\t")
            index = splitted[0]
            if not index.isdigit():
                continue
            word, pos, tag = splitted[WORD_COLUMN], splitted[POS_COLUMN], splitted[TAG_COLUMN]
            if pos == "PUNCT" and word in POS_MAPPING:
                pos = POS_MAPPING[word]
            if tag == "_":
                curr_sent.append(pos)
            else:
                curr_sent.append("{},{}".format(pos, tag))
            word = process_word(word, to_lower=to_lower, append_case=append_case)
            curr_word_sent.append(word)
            if pos == "_" and not has_errors:
                print(line)
                has_errors = True
        if len(curr_sent) > 0:
            answer.append((curr_word_sent, curr_sent))
    for i, (word_sent, tag_sent) in enumerate(answer):
        for j, (word, tag) in enumerate(zip(word_sent, tag_sent)):
            if attach_tokens:
                sep = "|" if "," in tag else ","
                word = "".join(word)
                if word in POS_MAPPING:
                    word = POS_MAPPING[word]
                tag_sent[j] += "{}token={}".format(sep, word)
    if not read_words:
        answer = [elem[1] for elem in answer]
    if wrap:
        return [[elem] for elem in answer]
    return answer

if __name__ == "__main__":
    L = len(sys.argv[1:])
    for i in range(0, L, 2):
        infile, outfile = sys.argv[1+i:3+i]
        answer = read_tags_infile(infile)
        with open(outfile, "w", encoding="utf8") as fout:
            for sent in answer:
                fout.write("\n".join(sent) + "\n\n")
