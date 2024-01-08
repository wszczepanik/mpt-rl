import ast
import json
import logging
import re
from pathlib import Path
from os import PathLike
from numbers import Number
from typing import Dict, List, Tuple, Type

import numpy as np
from ruamel.yaml import YAML

from pydantic import ValidationError

from sim_args import TcpRlSimArgs


class EnvParamGenerator:
    """Class used for QKDNetSim input file generation, yaml to json conversion.

    Allows usage of numpy.random.Generator methods instead of putting numbers. Most parameters have defaults. Provides checking.

    Allowed methods:
    ['_bit_generator', '_poisson_lam_max', 'beta', 'binomial', 'bit_generator', 'bytes', 'chisquare', 'choice', 'dirichlet', 'exponential', 'f', 'gamma', 'geometric', 'gumbel', 'hypergeometric', 'integers', 'laplace', 'logistic', 'lognormal', 'logseries', 'multinomial', 'multivariate_hypergeometric', 'multivariate_normal', 'negative_binomial', 'noncentral_chisquare', 'noncentral_f', 'normal', 'pareto', 'permutation', 'permuted', 'poisson', 'power', 'random', 'rayleigh', 'shuffle', 'standard_cauchy', 'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t', 'triangular', 'uniform', 'vonmises', 'wald', 'weibull', 'zipf']
    """

    def __init__(self, target_class: Type, input_file: str | PathLike = None) -> None:  # type: ignore
        """Constructor method

        Args:
            input_file (str | PathLike): Path to input YAML file.
            target_class (Type):
        """
        self.input_dict = dict()
        self.output = dict()

        self.target_class = target_class
        self.target_arguments = dict()
        if input_file:
            self.input_file = Path(input_file)
            self._load_yaml()
        self.rng = np.random.default_rng(0)

    def _load_yaml(self) -> None:
        """Read yaml file to dict.

        Returns:
            None
        """
        yaml = YAML(typ="safe")
        self.input_dict = yaml.load(self.input_file)["ns3Settings"]

    def generate(self) -> Dict:
        """Generate new parameters dictionary.

        Returns:
            Dict: target class as dictionary
        """

        self.target_arguments = dict()
        self.output = self.target_class(**self.target_arguments)

        for key, value in self.input_dict.items():
            # if is number then just copy number
            if isinstance(value, Number):
                self.target_arguments[key] = value
            # check if resembles function
            elif re.fullmatch(r"^[a-zA-Z0-9_]+\(.*\).*$", value):
                self.target_arguments[key] = self.generate_value(value)
            # otherwise hope that it is correct string argument
            else:
                self.target_arguments[key] = value

        self.output = self.target_class(**self.target_arguments)

        return self.output.asdict()

    # instance method because uses self.rng object
    def generate_value(self, func_string: str) -> int | float | str:
        """Generate value using method string from np.random.Generator.

        Args:
            func_string (str): function string

        Returns:
            int | float | str: generated scalar value
        """

        func, args, unit = self._validate_method_string(func_string)
        logging.debug(func, args)

        if args:
            generated = getattr(self.rng, func)(*args)
        else:
            generated = getattr(self.rng, func)()

        if isinstance(unit, str) and len(unit) > 0:
            generated = str(generated) + unit

        return generated

    @staticmethod
    def _validate_method_string(m_string: str) -> Tuple[str, Tuple, str]:
        """Check if format is correct: func(arg1,arg2)

        Validate if method exists and arguments are input correctly.
        Currently only ``*args`` are accepted, ``**kwargs`` does not work.

        Args:
            m_string (str): string of method with arguments ex. ``"integers(6, 21)"``

        Raises:
            TypeError: method not in dir(numpy.random.Generator)
            TypeError: wrong method format

        Returns:
            Tuple[str, Tuple, str]: parsed function name, arguments in tuple and optional unit
        """

        if not re.fullmatch(r"^[a-z_]+\([a-z0-9\.\, \[\]]*\)[a-zA-Z]*$", m_string):
            raise TypeError(
                f'"{m_string}" is in wrong format. Make sure it is a numpy.random.Generator method.'
            )

        f_name, args = m_string.split("(", 1)
        args, unit = args.rsplit(")", 1)

        # delete later
        print(f_name, args, unit)

        # check if generator has this method
        # todo: improve checking
        method_list = [
            method
            for method in dir(np.random.Generator)
            if method.startswith("__") is False
        ]
        if f_name not in method_list:
            raise TypeError(f"np.random.Generator does not have {f_name}() method.")

        # convert string to arguments
        args = ast.literal_eval(args)
        # fix when list is treated as multiple arguments instead of one
        # ex. choice([32, 64]) -> [32, 64] instead of ([32, 64],)
        if type(args) is not tuple:
            args = (args,)
        return f_name, args, unit


if __name__ == "__main__":
    INPUT_FILE = "example.yaml"

    generator = EnvParamGenerator(TcpRlSimArgs, INPUT_FILE)
    # seed == 0
    output = generator.generate()
    print(output)
