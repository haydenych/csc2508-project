import os

from datetime import datetime

class Logger:
    def __init__(self, dirname):
        self.dirname = dirname
        self.filename = os.path.join(
            self.dirname,
            datetime.now().strftime("%d%m%Y_%H%M%S") + ".txt"
        )

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        open(self.filename, "w").close()

    def log(self, msg=""):
        with open(self.filename, "a") as f:
            f.write(msg + "\n")