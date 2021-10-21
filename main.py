import issue_downloader
import numpy as np
import os
import re
import sklearn.svm
import sys
from sklearn import metrics, model_selection
from sklearn.feature_extraction import text
from typing import IO, List, Sequence, Optional


class Sample:
    """
    A single training example.

    Attributes:
        is_info_flow: If this is an info-flow bug description.
        data: The contents of the original bug description.
    """

    def __init__(self, is_info_flow: bool, data: str):
        self.is_info_flow = is_info_flow
        self.data = data


def read_file(file: IO) -> Sample:
    """
    Reads the contents of file into a Sample.

    Args:
        file: An open file containing a sample. Samples are formatted '[yes|no]\n[bug report]'.

    Returns:
        A sample representing the contents of the file.
    """
    first_line: str = file.readline().strip()
    data: str = file.read().replace('\n', ' ')
    return Sample(first_line == 'yes', data)


def read_directory(directory_name: str) -> Sequence[Sample]:
    """
    Reads the samples of directory into a sequence of Samples.

    The directory must be flat (no sub-directories) and only contain samples.

    Args:
        directory_name: The name of the directory containing the samples.

    Returns:
        A sequence of samples from the files in directory_name.
    """
    out = []
    for dirpath, _, filenames in os.walk(directory_name):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            with open(full_path, 'r') as file:
                out.append(read_file(file))
    return out


def preprocess(s: str) -> str:
    return re.sub(
        '[^a-zA-Z ]',
        '',
        # Remove domain names.
        re.sub(
            '.*\\.[a-zA-Z]+',
            '',
            # Remove XML comments.
            re.sub(
                '<!--.*-->',
                '',
                # Remove things enclosed in carrots.
                re.sub(
                    '<[^>]*>.*<[^>]*>',
                    '',
                    # Remove lines that start with spaces.
                    re.sub(
                        '^\s.*$',
                        '',
                        # Remove things enclosed in backticks.
                        re.sub('`.*`', '', s),
                    ),
                ),
            ),
        ),
    ).strip()


def remove_code(sample: Sample):
    """
    Removes all markdown source code from the sample.

    Args:
        sample: The sample to remove code from.
    """
    sample.data = preprocess(sample.data)


def exists(path: str) -> bool:
    try:
        os.stat(path)
        return True
    except:
        return False


def main(argv: Sequence[str]):
    if len(argv) != 5:
        print(
            f'usage: {argv[0]} [data directory] [repo owner] [repo name] [output directory]',
            file=sys.stderr,
        )
        exit(1)

    output_dir = argv[4]
    if not exists(output_dir):
        os.mkdir(output_dir)

    print('Step 1: Reading samples.')
    samples = read_directory(argv[1])
    for sample in samples:
        remove_code(sample)

    print('Step 2: TF-IDF vectorization.')
    tf_idf_vectorizer = text.TfidfVectorizer(
        input='content',
        strip_accents='unicode',
        stop_words='english',
    )
    X = tf_idf_vectorizer.fit_transform([sample.data for sample in samples])
    Y = np.array(
        [1 if sample.is_info_flow else -1 for sample in samples],
        dtype=int,
    )

    print('Step 3: Training SVM.')

    best_model = sklearn.svm.SVC()
    best_f1 = float('-inf')
    for i in range(0, 150):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            X,
            Y,
            test_size=0.4,
            random_state=i,
        )
        svm = sklearn.svm.SVC(
            kernel='linear',
        )
        svm.fit(x_train, y_train)

        f1 = metrics.f1_score(y_test, svm.predict(x_test))
        if f1 > best_f1:
            best_f1 = f1
            best_model = svm

    print(f'f1: {best_f1}')

    repo_owner = argv[2]
    repo_name = argv[3]
    print(f'Step 4: Looking at {repo_owner}/{repo_name}')

    bug_number = 0
    for issue in issue_downloader.download_issues(repo_owner, repo_name):
        if bug_number > 30:
            break
        try:
            input = preprocess(issue['body'])
            if len(input.split(' ')) < 10:
                print(f'skipping {issue["body"]} input = {input}')
                continue
            issue_vec = tf_idf_vectorizer.transform(input)
            prediction = best_model.predict(issue_vec)[0]
            print(prediction)
            if prediction == 1:
                print(f'saving to {output_dir}/{repo_name}_{issue["number"]}')
                with open(os.path.join(output_dir, repo_name + '_' + str(issue['number'])), 'w') as fd:
                    print(issue['body'], file=fd)
                    bug_number += 1
        except Exception:
            pass


if __name__ == '__main__':
    main(sys.argv)
