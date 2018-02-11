import sys

def descr_to_feats(symbol):
    if "," in symbol:
        symbol, feats = symbol.split(",")
        feats = tuple(tuple(x.split("=")) for x in feats.split("|"))
    else:
        feats = ()
    return symbol, feats

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
            # splitted = line.split(",")
            # feats = ()
            # if len(splitted) == 2:
            #     feats = tuple(tuple(x.split("=")) for x in splitted[1].split("|"))
            # curr_sent.append((splitted[0], feats))
            curr_sent.append(line)
        if len(curr_sent) > 0:
            answer.append([curr_sent])
    return answer


