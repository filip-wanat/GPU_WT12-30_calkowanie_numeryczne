class Printer:
    def __init__(self, filename, filetype):
        self.filename = filename
        self.filetype = filetype
        
    def __enter__(self):
        self.fd = open(f'{self.filename}.{self.filetype}', "w")
        return self

    def __exit__(self, type, value, traceback):
        self.fd.close()

    def print(self, value):
        self.fd.write(value)

    def printValues(self, *args):
        for a in args:
            self.fd.write(f"{a};")
        self.fd.write("\n")
