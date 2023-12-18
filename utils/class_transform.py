
class starToBinary:
    def __init__(self, mapping):
        self.mapping = mapping
    
    def map_values(self, values):
        return [self.mapping[value] for value in values]
    
# Example usage
# values = [0, 1, 3, 4, 2]
def map_values(values):
    mapper = starToBinary({0: 0, 1: 0, 2: 1, 3: 1, 4: 1})
    mapped_values = mapper.map_values(values)
    return mapped_values

