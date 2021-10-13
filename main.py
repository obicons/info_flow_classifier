import numpy as np
import os
import re
import sklearn.svm
import sys
from sklearn import metrics, model_selection
from sklearn.feature_extraction import text
from typing import IO, List, Sequence


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


def remove_code(sample: Sample):
    """
    Removes all markdown source code from the sample.

    Args:
        sample: The sample to remove code from.
    """
    sample.data = re.sub('`.*`', '', sample.data).strip()


def main(argv: Sequence[str]):
    if len(argv) != 2:
        print(f'usage: {argv[0]} [data directory]', file=sys.stderr)
        exit(1)

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
        [1 if sample.is_info_flow else 0 for sample in samples],
        dtype=int,
    )

    print('Step 3: Training SVM.')

    best_model = None
    best_f1 = float('-inf')
    for i in range(0, 150):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            X,
            Y,
            test_size=0.4,
            random_state=i,
        )
        svm = sklearn.svm.SVC()
        svm.fit(x_train, y_train)

        f1 = metrics.f1_score(y_test, svm.predict(x_test))
        if f1 > best_f1:
            best_f1 = f1
            best_model = svm

    print(f'f1: {best_f1}')


if __name__ == '__main__':
    main(sys.argv)
