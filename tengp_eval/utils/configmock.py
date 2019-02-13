class ConfigMock:
    def __init__(self):
        self.dicts = {}

    def __getitem__(self, key):
        if key not in self.dicts:
            self.dicts[key] = None
            return self.dicts[key]

        return self.dicts[key]

    def getint(self, dict_key, key):
        return int(self[dict_key][key])

    def getfloat(self, _, key):
        return float(self[dict_key][key])

    def add(self, key, value):
        self[dict_key][key] = value

