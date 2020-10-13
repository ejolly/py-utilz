"""
Piping class similar to %>% in R.
"""

class Pipe:

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __rshift__(self, other):
        if self.verbose:
            print("Pipe.rshift")
        if callable(other):
            self.data = other(self.data)
            return self.data
        elif isinstance(other, str):
            self.data = getattr(self.data, other).__call__()
            return self.data
        elif isinstance(other, tuple):
            methodname, *rest = other
            args, kwargs = [], {}
            for elem in rest:
                if isinstance(elem, dict):
                    kwargs = {**kwargs, **elem}
                else:
                    args.append(elem)
            if isinstance(methodname, str):
                self.data = getattr(self.data, methodname).__call__(*args, **kwargs)
            elif callable(methodname):
                self.data = methodname(self.data, *args, **kwargs)
            else:
                raise TypeError(
                    f"Received {type(other)}. Must pipe into a function or method"
                )
            return self.data
        elif isinstance(other, Pipe):
            return self.data
        else:
            raise TypeError(
                f"Received {type(other)}. Must pipe into a function or method"
            )

    def __gt__(self, other):
        if self.verbose:
            print("gt")
        self.__rshift__(other)
        return self.data

    def __rrshift__(self, other):
        if self.verbose:
            print(f"Pipe.rrshift ({type} fallback)")
        if callable(other):
            pass
        else:
            self.data = other
            return self
