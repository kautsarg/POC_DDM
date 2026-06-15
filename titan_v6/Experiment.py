from Well import Well

class Experiment:
    COLS = 204

    def __init__(self, wells_list: list[Well], exp_path_str=None):
        self.wells_list = wells_list
        self.exp_path_str = exp_path_str

    # build setter to check that wells_list is a list, and elements are instances of Well
    @property
    def wells_list(self):
        return self._wells_list

    @wells_list.setter
    def wells_list(self, value):
        if not isinstance(value, list):
            raise ValueError("Experiment.wells_list should be a list")
        for i in value:
            if not isinstance(i, Well):
                raise ValueError("The elements of Experiment.wells_list should be instances of Well")
        self._wells_list = value

    @property
    def nwells(self):
        return len(self.wells_list)

