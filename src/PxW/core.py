import prefect
from prefect import Task, Flow
from prefect.tasks.shell import ShellTask


WEKA = "/home/elia/Software/weka/weka-3-8-3/weka.jar"


class J48(object):

    prefixes = {
        "weka": "java -cp {}".format(WEKA),
        "J48": "weka.classifiers.trees.J48",
        "train": "t",
        "test": "T",
        "model": "d",
        "confidence_factor": "C",
        "min_samples_leaf": "M",
        "no_cv": "no-cv",
    }

    # Main Methods
    def __init__(
        self, confidence_factor=0.25, min_samples_leaf=2, no_cv=True, **kwargs
    ):

        self.algorithm = self.prefixes.get("J48")

        self.train_filename = kwargs.get("train_filename", None)
        self.test_filename = kwargs.get("test_filename", None)

        self.confidence_factor = confidence_factor
        self.min_samples_leaf = min_samples_leaf

        self.algorithm_command = """
        {} -{} {} -{} {}
        """.format(
            self.prefixes.get("J48"),
            self.prefixes.get("confidence_factor"),
            self.confidence_factor,
            self.prefixes.get("min_samples_leaf"),
            self.min_samples_leaf,
        )

        if no_cv:
            self.algorithm_command += " -{}".format(self.prefixes.get("no_cv"))

        return

    def fit(self, train_filename=None, model_filename=None, verbose=True):

        self.train_filename = train_filename
        self.model_filename = model_filename

        if self.model_filename is not None:
            self.model_command = """
            -{} {}
            """.format(
                self.prefixes.get("model"), self.model_filename
            )
        else:
            self.model_command = ""

        self.train_command = """
        - {} {}
        """.format(
            self.prefixes.get("train"), train_filename
        )

        command = """
        {} {} {} {}
        """.format(
            self.prefixes.get("weka"),
            self.algorithm_command,
            self.train_command,
            self.model_command,
        )

        shell = ShellTask()

        with Flow("fit") as f:
            fit = shell(command=command)

        status = f.run()

        if verbose:
            print(status.result[fit].result.decode("utf-8"))

        return status

    def predict(self, **kwargs):
        return
