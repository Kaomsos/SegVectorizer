from collections import UserDict


class SemiIdentityMapping(UserDict):
    """
     a dict that should not be iterated
     which defines rules of a mapping
    """
    def __getitem__(self, item):
        key = item

        while key in self.data.keys():
            # to avoid infinite loop
            if key == self.data[key]:
                break
            key = self.data[key]

        return key

    def __call__(self, x):
        return self.__getitem__(x)