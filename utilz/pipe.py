class Pipe:
    """
    Piping class similar to %>% in R. Doesn't currently work with numpy arrays
    
    Usage:
    
    # 1) First create the pipe object at the top of your code and name it whatever you want. I like 'o'
    
    o = Pipe()
    
    # 2) Then use it with the '>>o>>' syntax
    
    # Some data to work with
    from seaborn import load_dataset
    df = load_dataset('iris')
    
    # a) pipe to another function, including lambdas
    
    df >>o>> print
    
    df >>o>> (lambda df: df*2)
    
    # b) pipe to a method and call it as a string without the '.' (must exist on the inputted object)
    
    df >>o>> 'head'

    # c) pass args and kwargs to the method or function as a tuple. The first item in the tuple should be the method name (str) or function (callable). Subsequent items are interpreted as kwargs if they're dicts, or args if they're anything else.
    
    df >>o>> ('mean', 1)
    # equivalent to df.mean(1)
    
    df >>o>> ('mean', 1, {'numeric_only': True})
    # df.mean(1,numeric_only=True)

    df >>o>> (pd.melt, {'id_vars': 'species'}, {'value_vars': 'petal_length'})
    # pd.melt(df, id_vars='species', value_vars='petal_length')

    # below is same as above since melt is both a method on DataFrames and a module function in pandas
    
    df >>o>> ('melt', {'id_vars': 'species'}, {'value_vars': 'petal_length'})
    # df.melt(id_vars='species', value_vars='petal_length')

    # You can also combine kwargs into a single dict
    df >>o>> (pd.melt, {'id_vars': 'species', 'value_vars': 'petal_length'})
    
    
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __rshift__(self, other):
        if self.verbose:
            print('Pipe.rshift')
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
                raise TypeError(f"Received {type(other)}. Must pipe into a function or method")
            return self.data
        elif isinstance(other, Pipe):
            return self.data
        else:
            raise TypeError(f"Received {type(other)}. Must pipe into a function or method")

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
