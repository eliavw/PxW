from io import StringIO

import pandas as pd
import prefect
from prefect import Flow, Task
from prefect.tasks.shell import ShellTask

# WEKA = "/home/elia/Software/weka/weka-3-8-3/weka.jar"
WEKA = "/home/zissou/Software/weka/weka.jar"


class J48(object):

    prefixes = {
        "weka": "java -cp {}".format(WEKA),
        "J48": "weka.classifiers.trees.J48",
        "train": "t",
        "test": "T",
        "dump_model": "d",
        "load_model": "l",
        "confidence_factor": "C",
        "min_samples_leaf": "M",
        "no_cv": "no-cv",
        "predict": "p",
    }

    # Main Methods
    def __init__(
        self, confidence_factor=0.25, min_samples_leaf=2, no_cv=True, **kwargs
    ):

        self.algorithm = self.prefixes.get("J48")
        self.model_filename = "tree.model"  # Default value

        self.confidence_factor = confidence_factor
        self.min_samples_leaf = min_samples_leaf

        self.algorithm_command = """{} -{} {} -{} {}""".format(
            self.prefixes.get("J48"),
            self.prefixes.get("confidence_factor"),
            self.confidence_factor,
            self.prefixes.get("min_samples_leaf"),
            self.min_samples_leaf,
        )

        if no_cv:
            self.algorithm_command += " -{}".format(self.prefixes.get("no_cv"))

        return

    def fit(self, train_filename, model_filename=None, verbose=True):

        self.train_filename = train_filename

        if model_filename is not None:
            self.model_filename = model_filename

        self.model_command = """-{} {}""".format(
            self.prefixes.get("dump_model"), self.model_filename
        )

        self.train_command = """-{} {}""".format(
            self.prefixes.get("train"), self.train_filename
        )

        self.command = """{} {} {} {}""".format(
            self.prefixes.get("weka"),
            self.algorithm_command,
            self.train_command,
            self.model_command,
        )

        shell = ShellTask()

        with Flow("fit") as f:
            fit = shell(command=self.command)

        status = f.run()

        if verbose and status.is_successful():
            print(status.result[fit].result.decode("utf-8"))

        return status

    def predict(self, test_filename, prediction_filename=None, verbose=True, **kwargs):

        self.test_filename = test_filename

        if prediction_filename is not None:
            self.prediction_filename = prediction_filename
        else:
            self.prediction_filename = 0  # Weka default

        self.model_command = """-{} {}""".format(
            self.prefixes.get("load_model"), self.model_filename
        )

        self.test_command = """-{} {}""".format(
            self.prefixes.get("test"), self.test_filename
        )

        self.predict_command = """-classifications weka.classifiers.evaluation.output.prediction.CSV"""

        self.command = """{} {} {} {} {}""".format(
            self.prefixes.get("weka"),
            self.prefixes.get("J48"),
            self.test_command,
            self.model_command,
            self.predict_command,
        )

        shell = ShellTask()

        with Flow("predict") as f:
            predict = shell(command=self.command)

        status = f.run()

        if verbose and status.is_successful():
            output = status.result[predict].result.decode("utf-8")

            df = self._parse_output(output)

            return df
        else:
            return status

    @staticmethod
    def _parse_output(output):
        df = pd.read_csv(
            StringIO(output),
            index_col=0,
            header=1,
            usecols=[0, 1, 2],
            lineterminator="\n",
        )
        return df
