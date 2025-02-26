from typing import Type

import gdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import re

from numpy.core.defchararray import add

ARR_TYPE_REGEX = r"(.*) \[([0-9]*)\]"
ARR_PTR_TYPE_REGEX = r"(.*) \(\*\)\[([0-9]*)\]"
PTR_TYPE_REGEX = r"(.*) \*"


class NoPointerSizeException(Exception):
    pass


class ArrayParseException(Exception):
    pass


_type_list: dict[str, type[np.dtype]] = {
    # C types
    "char": np.int8,
    "unsigned char": np.uint8,
    "short": np.int16,
    "unsigned short": np.uint16,
    "int": np.int32,
    "unsigned int": np.uint32,
    "float": np.float32,
    "double": np.float64,
    # Rust types
    "u8": np.uint8,
    "i8": np.int8,
    "i16": np.int16,
    "u16": np.uint16,
    "i32": np.int32,
    "u32": np.uint32,
    "usize": np.uint32,
    "f32": np.float32,
    "f64": np.float64,
}

_colors = ["r", "b", "g", "c", "m", "y"]


def parse_maybe_array(
    val: gdb.Value,
    candidate_size_arg: str = "",
) -> tuple[list, str]:
    """
    Get the underlying array and value type

    `val` should be of type array, or pointer to array, or raw pointer
    If it is a raw pointer, a size argument should have been passed to
    the command: pass it via `candidate_size_arg`
    """
    val_type = str(val.type)

    array = []
    array_size = None
    array_type = None

    print(val_type)
    match_arr = re.match(ARR_TYPE_REGEX, val_type)
    match_ptr_to_arr = re.match(ARR_PTR_TYPE_REGEX, val_type)
    match_ptr = re.match(PTR_TYPE_REGEX, val_type)
    if match_arr is not None:
        print("match arr")
        array_type = match_arr.group(1)
        array_size = int(match_arr.group(2))
    if match_ptr_to_arr is not None:
        print("match ptr arr")
        val = val.dereference()
        array_type = match_ptr_to_arr.group(1)
        array_size = int(match_ptr_to_arr.group(2))
    if match_ptr is not None:
        print("match ptr")
        array_type = match_ptr.group(1)
        try:
            array_size = gdb.parse_and_eval(candidate_size_arg)
            assert str(array_size.type) in (
                "int",
                "long",
                "long long",
                "unsigned",
                "unsigned long",
                "unsigned long long",
            )
            array_size = int(array_size)
        except Exception as e:
            raise NoPointerSizeException() from e
    if array_size is not None and array_type is not None:
        for i in range(array_size):
            array.append(_type_list[array_type](val[i]))

        return array, array_type
    else:
        raise ArrayParseException()


class ProcessPlotter(object):
    def __init__(self):
        pass

    def terminate(self):
        plt.close("all")

    def call_back(self):
        has_data = False
        while self.pipe.poll():
            plot_data = self.pipe.recv()
            has_data = True
            if plot_data is None:
                self.ax.cla()
            else:
                self.ax.plot(
                    plot_data["X"],
                    plot_data["Y"],
                    color=plot_data["color"],
                    label=plot_data["label"],
                )

        if has_data:
            self.ax.legend()
            self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print("starting plotter...")

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()

        timer = self.fig.canvas.new_timer(interval=1000)
        timer.add_callback(self.call_back)
        timer.start()

        plt.show()


class Plot(gdb.Command):
    """Plots array passed as argument (variable name)"""

    def __init__(self):
        super(Plot, self).__init__("plot", gdb.COMMAND_USER)
        self.plot_pipe, self.plotter_pipe = mp.Pipe()
        self.plotter = None
        self.plot_process = None

    def invoke(self, argument, from_tty):
        # Check args
        if len(argument) == 0:
            print("Wrong arguments, pass one or several variables")
            return

        # Start one side process (and only one) to display data
        if self.plotter == None:
            self.plotter = ProcessPlotter()
            self.plot_process = mp.Process(
                target=self.plotter, args=(self.plotter_pipe,), daemon=True
            )
            self.plot_process.start()
        elif not self.plot_process.is_alive():
            self.plotter = ProcessPlotter()
            self.plot_process = mp.Process(
                target=self.plotter, args=(self.plotter_pipe,), daemon=True
            )
            self.plot_process.start()

        # Parse arguments
        args = argument.split(" ")

        # Reset the current graph by sending a None message to subprocess
        self.plot_pipe.send(None)

        i = 0
        while i < len(args):
            val = gdb.parse_and_eval(args[i])
            val_type = str(val.type)

            # Init variables for each new variables
            array = []
            array_name = args[i]
            array_size = None
            array_type = None
            address = 0

            # If variable is reference to array
            #  - dereference it first
            #  - get type and size
            match_arr = re.match(ARR_TYPE_REGEX, val_type)
            match_ptr_to_arr = re.match(ARR_PTR_TYPE_REGEX, val_type)
            match_ptr = re.match(PTR_TYPE_REGEX, val_type)
            if match_arr is not None:
                array_type = match_arr.group(1)
                array_size = int(match_arr.group(2))
                address = val.address
            if match_ptr_to_arr is not None:
                val = val.dereference()
                array_type = match_ptr_to_arr.group(1)
                array_size = int(match_ptr_to_arr.group(2))
                address = val.address
            if match_ptr is not None:
                if i == len(args) - 1:
                    print(
                        f"Variable '{args[i]}' is a pointer, but you didn't pass a size as a 2nd argument"
                    )
                    i += 1
                    return
                array_type = match_ptr.group(1)
                try:
                    array_size = int(gdb.parse_and_eval(args[i + 1]))
                    address = val
                    i += 1
                except Exception:
                    print(
                        f"Error: variable '{args[i]}' is a pointer, but following argument is not an integer"
                    )
                    i += 1
                    return
                print(f"Variable '{array_name}' is a pointer, using user-supplied size")
            # If everything went well and the variable can be parsed
            #  - parse it
            if array_size is not None and array_type is not None and address != 0:
                print(
                    "Parsing {}; array of size {}, type {}, array data starting at {}".format(
                        array_name, array_size, array_type, hex(int(address))
                    )
                )

                array = np.frombuffer(
                    gdb.selected_inferior().read_memory(
                        address,
                        array_size * _type_list[array_type]().itemsize,
                    ),
                    dtype=_type_list[array_type],
                    count=-1,
                )
                # Send to subprocess :
                #  - array data
                #  - color for pretty printing
                #  - variable name for caption
                self.plot_pipe.send(
                    {
                        "X": range(len(array)),
                        "Y": array,
                        "color": _colors[i % len(_colors)],
                        "label": array_name,
                    }
                )
            else:
                print(
                    "Variable '{}' is not an array or cannot be parsed currently".format(
                        args[i]
                    )
                )
            i += 1


class PlotXY(gdb.Command):
    """Plot array Y = f(X) passed as arguments"""

    def __init__(self):
        super(PlotXY, self).__init__("plotxy", gdb.COMMAND_USER)
        self.plot_pipe, self.plotter_pipe = mp.Pipe()
        self.plotter = None
        self.plot_process = None

    def invoke(self, argument, from_tty):
        # Check args
        if len(argument) == 0:
            print("Wrong arguments, pass one or several variables")
            return

        # Start one side process (and only one) to display data
        if self.plotter == None:
            self.plotter = ProcessPlotter()
            self.plot_process = mp.Process(
                target=self.plotter, args=(self.plotter_pipe,), daemon=True
            )
            self.plot_process.start()
        elif not self.plot_process.is_alive():
            self.plotter = ProcessPlotter()
            self.plot_process = mp.Process(
                target=self.plotter, args=(self.plotter_pipe,), daemon=True
            )
            self.plot_process.start()

        # Parse arguments
        args: list[str] = argument.split(" ")

        # Reset the current graph by sending a None message to subprocess
        self.plot_pipe.send(None)

        if len(args) not in (2, 3):
            print(f"Expected 2 or 3 arguments (X, Y(, size)), but got {len(args)}")
            return
        X = gdb.parse_and_eval(args[0])
        try:
            X_values, X_element_type = parse_maybe_array(
                X,
                args[-1] if args[-1].isnumeric() else "",
            )
        except Exception as e:
            if isinstance(e, NoPointerSizeException):
                print(
                    f"X is of pointer type but no size argument was specified."
                    f"When using pointer types, usage is: `plotxy <X> <Y> <number of elements>`"
                )
                return
            if isinstance(e, ArrayParseException):
                print(f"Could not parse X as an array; type is '{str(X.type)}'")
                return
            else:
                print(f"Unexpected exception: '{e}'")
                return
        print(f"X: array of type '{X_element_type}' of length {len(X_values)}")
        Y = gdb.parse_and_eval(args[1])
        try:
            Y_values, Y_element_type = parse_maybe_array(
                Y,
                args[-1] if args[-1].isnumeric() else "",
            )
        except Exception as e:
            if isinstance(e, NoPointerSizeException):
                # technically, we can guess the size using that of X, but it's more
                # consistent and intuitive to ask for size
                print(
                    f"Y is of pointer type but no size argument was specified."
                    f"When using pointer types, usage is: `plotxy <X> <Y> <number of elements>`"
                )
                return
            if isinstance(e, ArrayParseException):
                print(f"Could not parse Y as an array; type is '{str(Y.type)}'")
                return
            else:
                print(f"Unexpected exception: '{e}'")
                return
        print(f"Y: array of type '{Y_element_type}' of length {len(Y_values)}")

        if len(X_values) != len(Y_values):
            print(f"Aborting plot: mismatching lengths !")
            return

        self.plot_pipe.send(
            {
                "X": X_values,
                "Y": Y_values,
                "color": "tab:blue",
                "label": f"{args[1]} = f({args[0]})",
            }
        )


Plot()
PlotXY()
