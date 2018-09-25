from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

# base sentences
s = "Bill has never done anything terrible."
# s =  "Exactly half had never done anything terrible."

# some sample sentences of the two licensing contexts with different subjects
# s_names = "Bill has not done anything terrible. Sally has not done anything terrible. John has not done anything terrible. Isaac has not done anything terrible. I have not done anything terrible."
# s_numbers = "Exactly half had never done anything terrible. Exactly three-quarters had never done anything terrible. Exactly ten had never done anything terrible. Exactly ninety percent had never done anything terrible. Exactly 60% had never done anything terrible."

# replaces not with never and replaces modifier - change as necessary
# s= s_names.replace("not", "never").replace("terrible", "cynical")
# s= s_numbers.replace("never", "not").replace("terrible", "cynical")


# for multiple sentences, split on period
# sents = s.split(".")

# use this for a single sentence
sents = [s]

for a in sents:
    res = nlp.annotate(a,
                       properties={
                           'annotators': 'sentiment',
                           'outputFormat': 'json',
                           'timeout': 1000,
                       })

    for s in res["sentences"]:
        print("%d: '%s': %s %s" % (
            s["index"],
            " ".join([t["word"] for t in s["tokens"]]),
            s["sentimentValue"], s["sentiment"]))
        start = s["sentimentTree"].find("prob=")
        t = s["sentimentTree"][start:]
        end = t.find(" ")
        print(s["sentimentTree"][start:start+end])

        # uncomment to print parse tree
        # print(s["parse"])