class HitSetGenerativeModel:
    """
    Model for generating hit sets at time t+1 given the hit set at time t.
    """

    def __init__(self, encoder, size_generator, set_generator):
        self.encoder = encoder
        self.size_generator = size_generator
        self.set_generator = set_generator
