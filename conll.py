import re
import tempfile
import subprocess
import operator
import collections
import logging

logger = logging.getLogger(__name__)

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")  # First line at each document
COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)


def get_doc_key(doc_id, part):
    return "{}_{}".format(doc_id, int(part))


def output_conll(input_file, output_file, predictions, subtoken_map):
    prediction_map = {}
    for doc_key, clusters in predictions.items():
        start_map = collections.defaultdict(list)
        end_map = collections.defaultdict(list)
        word_map = collections.defaultdict(list)
        for cluster_id, mentions in enumerate(clusters):
            for start, end in mentions:
                start, end = subtoken_map[doc_key][start], subtoken_map[doc_key][end]
                if start == end:
                    word_map[start].append(cluster_id)
                else:
                    start_map[start].append((cluster_id, end))
                    end_map[end].append((cluster_id, start))
        for k,v in start_map.items():
            start_map[k] = [cluster_id for cluster_id, end in sorted(v, key=operator.itemgetter(1), reverse=True)]
        for k,v in end_map.items():
            end_map[k] = [cluster_id for cluster_id, start in sorted(v, key=operator.itemgetter(1), reverse=True)]
        prediction_map[doc_key] = (start_map, end_map, word_map)

    filtered_docs = set()

    word_index = 0
    for line in input_file.readlines():
        row = line.split()
        if len(row) == 0:
            output_file.write("\n")
        elif row[0].startswith("#"):
            begin_match = re.match(BEGIN_DOCUMENT_REGEX, line)
            if begin_match:
                doc_key = get_doc_key(begin_match.group(1), begin_match.group(2))
                if doc_key not in prediction_map:
                    filtered_docs.add(doc_key)
                    continue
                start_map, end_map, word_map = prediction_map[doc_key]
                word_index = 0
            output_file.write(line)
            output_file.write("\n")
        else:
            assert get_doc_key(row[0], row[1]) == doc_key
            if doc_key in filtered_docs:
                continue
            coref_list = []
            if word_index in end_map:
                for cluster_id in end_map[word_index]:
                    coref_list.append("{})".format(cluster_id))
            if word_index in word_map:
                for cluster_id in word_map[word_index]:
                    coref_list.append("({})".format(cluster_id))
            if word_index in start_map:
                for cluster_id in start_map[word_index]:
                    coref_list.append("({}".format(cluster_id))

            if len(coref_list) == 0:
                row[-1] = "-"
            else:
                row[-1] = "|".join(coref_list)

            output_file.write("   ".join(row))
            output_file.write("\n")
            word_index += 1

def official_conll_eval(gold_path, predicted_path, metric, official_stdout=True):
    # Replace this path with your local copy of perl.
    PERL_PATH = r"C:\Users\Ofir\anaconda3\envs\nlp2\Library\bin\perl.exe"
    cmd = [PERL_PATH, "reference-coreference-scorers/scorer.pl", metric, gold_path, predicted_path, "none"]
    logger.info("Running official conll eval: {}".format(" ".join(cmd)))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    res = process.wait()
    assert res == 0, f"failed with {stdout}"

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        logger.error(stderr)

    if official_stdout:
        logger.info("Official result for {}".format(metric))
        logger.info(stdout)

    coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
    recall = float(coref_results_match.group(1))
    precision = float(coref_results_match.group(2))
    f1 = float(coref_results_match.group(3))
    return {"r": recall, "p": precision, "f": f1}


def evaluate_conll(gold_path, predictions, subtoken_maps, official_stdout=True):
    predictions_temp = "prediction.remove.me"
    with open(predictions_temp, mode="w", encoding="utf-8") as prediction_file:
        with open(gold_path, "r", encoding="utf-8") as gold_file:
            output_conll(gold_file, prediction_file, predictions, subtoken_maps)
        # logger.info("Predicted conll file: {}".format(prediction_file.name))
    results = {m: official_conll_eval(gold_path, predictions_temp, m, official_stdout) for m in ("muc", "bcub", "ceafe") }
    return results
