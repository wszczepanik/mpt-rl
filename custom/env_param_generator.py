import ast
import json
import logging
import re
from pathlib import Path
from os import PathLike
from numbers import Number
from typing import Dict, List, Type

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

    def __init__(
        self,
        input_file: str | PathLike,
        target_class: Type
    ) -> None:
        """Constructor method

        Args:
            input_file (str | PathLike): Path to input YAML file.
            target_class (Type):
        """
        self.input_yaml = None
        self.output = dict()

        self.target_class = target_class
        self.input_file = Path(input_file)
        self.load_yaml()

    def load_yaml(self) -> None:
        """Read yaml file to dict.

        Returns:
            None
        """
        yaml = YAML(typ="safe")
        self.input_yaml = yaml.load(self.input_file)

    def generate(self) -> Dict:
        """Generate new QKDNetSim config and save to file.

        Returns:
            Dict: target class as dictionary
        """

        self.output = self.target_class()
        # # kms_nodes is required, only one can exist
        # self.output_dict["kms_nodes"] = dict()
        # for key, value in self.input_yaml["kms_nodes"].items():
        #     # if is number then just copy
        #     if isinstance(value, Number):
        #         self.output_dict["kms_nodes"][key] = value
        #     # try generating number using numpy generator
        #     else:
        #         self.output_dict["kms_nodes"][key] = self.generate_value(value)

        # # everything else is optional so check if it was in config
        # if self.input_yaml["qkd_links"]:
        #     self.output_dict["qkd_links"] = self.loop_generate_dict(
        #         self.input_yaml["qkd_links"]
        #     )

        # # check if output is in correct format
        # # it is simpler to check at completion
        # try:
        #     self.validate_output()
        # except ValidationError as e:
        #     logging.error(e)

        return self.output.asdict()

    @staticmethod
    def loop_generate_dict(self, in_dict: dict | List[dict]) -> List[dict]:
        """Generate one type of object config using dict or list of dicts.

        Args:
            in_dict (dict | List[dict]): Config dictionary or list of config dictionaries

        Returns:
            List[dict]: list of dictionaries with config accepted by QKDNetSim
        """
        out_dicts = list()
        # if more than one to create (list of dicts)
        if type(in_dict) is list:
            for in_dict_element in in_dict:
                out_dicts.append(dict())
                for key, value in in_dict_element.items():
                    # if is number then just copy
                    if isinstance(value, Number):
                        out_dicts[-1][key] = value
                    # try generating number using numpy generator
                    else:
                        out_dicts[-1][key] = self.generate_value(value)
        # else create one (dict)
        else:
            out_dicts.append(dict())
            for key, value in in_dict:
                # if is number then just copy
                if isinstance(value, Number):
                    out_dicts[-1][key] = value
                # try generating number using numpy generator
                else:
                    out_dicts[-1][key] = self.generate_value(value)
        return out_dicts

    # # instance method because uses self.rng object
    # def generate_value(self, func_string: str) -> int | float:
    #     """Generate value using method string from np.random.Generator.

    #     Args:
    #         func_string (str): function string

    #     Returns:
    #         int | float: generated scalar value
    #     """

    #     # func, args = self._validate_method_string(func_string)
    #     # logging.debug(func, args)

    #     # logging.debug(f"{generated=}")  # prints variable=value

    #     # # json is not compatible with numpy types, so convert them
    #     # # sometimes it may be interpreted as python type (like float)
    #     # try:
    #     #     json.dumps(generated)
    #     #     return generated
    #     # # sometimes it returns numpy type (like int64)
    #     # except:
    #     #     return generated.item()

    # @staticmethod
    # def _validate_method_string(m_string: str) -> (str, list):
    #     """Check if format is correct: func(arg1,arg2)

    #     Validate if method exists and arguments are input correctly.
    #     Currently only ``*args`` are accepted, ``**kwargs`` does not work.

    #     Args:
    #         m_string (str): string of method with arguments ex. ``"integers(6, 21)"``

    #     Raises:
    #         TypeError: method not in dir(numpy.random.Generator)
    #         TypeError: wrong method format

    #     Returns:
    #         str: method name
    #         list: parsed method arguments
    #     """

    #     if re.fullmatch("[a-z]+\([a-z0-9\.\, \[\]]*\)$", m_string):
    #         # strip from ")"
    #         m_string = m_string[:-1]
    #         # split function name
    #         splitted = re.split("[\(\)]", m_string)
    #         f_name = splitted[0]
    #         # check if generator has this method
    #         # todo: improve checking
    #         method_list = [
    #             method
    #             for method in dir(np.random.Generator)
    #             if method.startswith("__") is False
    #         ]
    #         if f_name not in method_list:
    #             raise TypeError(f"np.random.Generator does not have {f_name}() method.")

    #         # take only arguments string
    #         splitted = splitted[1]
    #         # convert string to arguments
    #         splitted = ast.literal_eval(splitted)
    #         # fix when list is treated as multiple arguments instead of one
    #         # ex. choice([32, 64]) -> [32, 64] instead of ([32, 64],)
    #         if type(splitted) is not tuple:
    #             splitted = (splitted,)
    #         return f_name, splitted
    #     else:
    #         raise TypeError(
    #             f'"{m_string}" is in wrong format. Make sure it is a numpy.random.Generator method.'
    #         )

    def validate_output(self) -> None:
        """Throw exception if field is missing or wrong type.

        Output is validated using schemas from qkd_schema.
        """

        # data = self.output_dict
        # # check if required field exists
        # if "kms_nodes" in data:
        #     # currently only one pair is supported
        #     qkd_schema.KMSNodesModel(**data["kms_nodes"])
        # else:
        #     raise TypeError("kms_nodes field is required.")
        # # check if exists
        # if "qkd_links" in data:
        #     # check if more than one exist
        #     if isinstance(data["qkd_links"], list):
        #         for link in data["qkd_links"]:
        #             qkd_schema.QKDLinksModel(**link)
        #     else:
        #         qkd_schema.QKDLinksModel(data["qkd_links"])


if __name__ == "__main__":
    INPUT_FILE = "example.yaml"

    generator = EnvParamGenerator(INPUT_FILE, TcpRlSimArgs)
    # generator.load()
    output = generator.generate()
    print(output)
