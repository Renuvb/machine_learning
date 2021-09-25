
class Layer(object):
    def __init__(self, input_shape=(None, 1), activation=None):
        self.input_shape = input_shape
        self.output_shape = (None, 0)
        self.current_input = None
        self.current_output = None
        self.w = None
        self.current_loss = None
        self.eta = 0.1
        if activation != None:
            self.activation = activation()
        else:
            self.activation = None

    def build(self):
        pass

    def do_forward(self, input):
        self.current_input = input
        self.current_output = self.forward(input)
        if self.activation is None:
            return self.current_output
        else:
            return self.activation.forward(self.current_output)

    def do_backward(self, sensitive):
        if self.activation is None:
            return self.backward(sensitive)
        else:
            return self.backward(self.activation.backward(sensitive))

    def forward(self, input):
        pass

    def backward(self, sensitive):
        pass

    def input_shape(self):
        return self.input_shape

    def output_shape(self):
        return self.output_shape

    def loss(self):
        return self.current_loss