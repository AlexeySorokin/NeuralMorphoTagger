import sys

def descr_to_feats(symbol):
    if "," in symbol:
        symbol, feats = symbol.split(",", maxsplit=1)
        fields = []
        for elem in feats.split("|"):
            key, values = elem.split("=", maxsplit=1)
            values = values.split(",")
            fields.extend((key, value) for value in values)
        fields = tuple(fields)
    else:
        fields = ()
    return symbol, fields

def read_tags_input(infile):
    answer, curr_sent = [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                if len(curr_sent) > 0:
                    answer.append([curr_sent])
                curr_sent = []
                continue
            curr_sent.append(line)
        if len(curr_sent) > 0:
            answer.append([curr_sent])
    return answer


