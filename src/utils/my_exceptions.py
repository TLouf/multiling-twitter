class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, options):
        self.expression = expression
        self.message = (
            "{} isn't a valid input, available options are {}.".format(
                expression,options))

    def __str__(self):
        return self.message
