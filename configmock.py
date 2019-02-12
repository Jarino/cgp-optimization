class ConfigMock:
    def __init__(self):
        self.parameters = {}

    def getint(self, _, key):
        return int(self.parameters[key])

    def getfloat(self, _, key):
        return float(self.parameters[key])

    def add(self, key, value):
        self.parameters[key] = value

