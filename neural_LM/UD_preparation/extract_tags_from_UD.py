import sys

WORD_COLUMN, POS_COLUMN, TAG_COLUMN = 1, 3, 5

POS_MAPPING = {".": "<SENT>", "?": "<QUESTION>", "!":"<EXCLAM>",
               ",": "<COMMA>", "-": "<HYPHEN>", "--": "<DASH>",
               ":": "COLON", ";": "SEMICOLON", "\"": "<QUOTE>"}

if __name__ == "__main__":
    L = len(sys.argv[1:])
    for i in range(0, L, 2):
        infile, outfile = sys.argv[1+i:3+i]
        curr_sent, answer = [], []
        with open(infile, "r", encoding="utf8") as fin:
            print(infile)
            has_errors = False
            for line in fin:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if line == "":
                    if len(curr_sent) > 0:
                        answer.append(curr_sent)
                    curr_sent = []
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
                if pos == "_" and not has_errors:
                    print(line)
                    has_errors = True
        if len(curr_sent) > 0:
            answer.append(curr_sent)
        with open(outfile, "w", encoding="utf8") as fout:
            for sent in answer:
                fout.write("\n".join(sent) + "\n\n")
