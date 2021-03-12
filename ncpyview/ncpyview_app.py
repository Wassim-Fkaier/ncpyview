#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:27:08 2020
This module creates an interactive aplication using the streamlit framework
"""
__author__ = "Wassim Fkaier"
__maintainer__ = "Wassim Fkaier"
__email__ = "Wassim Fkaier@outlook.com"
__status__ = "Dev"
__package_name__ = "ncpyview"

import os
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import xarray
import plotly.express as plotly_express
import plotly.graph_objects as go
import json
import pkg_resources
from .readparam import parserarg


@st.cache(hash_funcs={xarray.core.dataset.Dataset: id}, allow_output_mutation=True)
def readnc(filename):
    """
    This function reads a netcdf file and returns it as a Dataset object.

    Parameters
    ----------
    filename : str
        The file 's name.

    Returns
    -------
    xarray.open_dataset(filename) : xarray.core.dataset.Dataset
        The Dataset object readed from the netcdf file.
    """

    return xarray.open_dataset(filename, decode_times=False).load()


class Ncpyviewer:
    """
    This class method reads data from netcdf, calculates the statistics
    and visualize them.
    """

    configfile_path = pkg_resources.resource_filename(
        __package_name__, os.path.join("etc", "config.json")
    )
    # Dictionnary which contains functions to compute statistics
    dict_funcs = {
        "min": lambda dataset, axis: np.nanmin(dataset, axis=axis),
        "max": lambda dataset, axis: np.nanmax(dataset, axis=axis),
        "mean": lambda dataset, axis: np.nanmean(dataset, axis=axis),
        "std": lambda dataset, axis: np.nanstd(dataset, axis=axis),
    }

    def __init__(self, dataset):
        """
        Initialize attributes

        Parameters
        ----------
        files_names : dict
            A dictionnary which contanins the netcdf files paths 
            or Bytes objects 
        dataset : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        self.dataset = dataset

    @property
    def dataset(self):
        """Getter"""
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset):
        """Setter"""
        if not list(dataset.variables):
            raise ValueError("Empty dataset")
        if not hasattr(dataset, "dims"):
            raise ValueError("The dataset has not the attribute dims")
        if not list(dataset.dims):
            raise ValueError("The dataset has no dimesnions")
        self.__dataset = dataset

    @classmethod
    def open_dataset(cls):
        """
        Classmethod to construct an instance object of type Ncpyviewer.
        It uploads files from the disk and can also parses the files paths
        from the command line as an arguments of the program
        and returns an instance of the Ncpyviewer class object.

        Parameters
        ----------
        cls : Ncpyviewer
            Ncpyviewer class object.

        Returns
        -------
        Ncpyviewer
            An instance object of the Ncpyviewer class object.

        """

        # Try to parse arguments from the command line
        args = parserarg.parsearg_func()
        files_from_cmd = args["ncfiles"]
        # Request to upload files
        ncfiles = st.file_uploader(
            "Choose a netcdf file(s)", accept_multiple_files=True
        )
        # if no file path has been parsed neither uploaded then stop flow
        if (ncfiles is None or not ncfiles) and not files_from_cmd:
            st.stop()
        # Files uploaded
        if (isinstance(ncfiles, list) or isinstance(ncfiles, tuple)) and ncfiles:
            # create a dictionnary
            ncfiles = {str(ncfile.name): ncfile for ncfile in ncfiles}
        # if no file has been uploaded
        if not ncfiles:
            # initialize a dict
            ncfiles = {}
        # at least a file path hase been parsed
        if files_from_cmd is not None:
            for file_path in files_from_cmd:
                ncfiles[file_path.split("/")[-1]] = file_path
        # Select a file and
        # Return an instance of the class object Ncpyviewer
        return cls(
            readnc(ncfiles[st.selectbox("Pick a file", options=list(ncfiles.keys()))])
        )

    @staticmethod
    def general_presentation():
        """
        A general presentation

        Returns
        -------
        None.

        """

        st.sidebar.title("**:panda_face: ncpyview app:panda_face:**")
        st.sidebar.date_input("Date", datetime.date.today())

    def read_config(self):
        """
        Read config file

        Returns
        -------
        None.

        """
        with open(self.configfile_path, "r") as config_file:
            self.configfile = json.load(config_file)

    def show_file_attributes(self):
        """
        This method renders a table which contains the global file attributes

        Returns
        -------
        None.

        """
        # Global attributes
        st.header("**Global file attributes**")
        # Get all the global attributes of the file and show them in a table
        st.table(
            pd.DataFrame(
                data={attr: [self.dataset.attrs[attr]] for attr in self.dataset.attrs}
            )
        )

    def select_variable(self):
        """
        This method selects a variable from all the dataset 's variables

        Returns
        -------
        None.

        """

        self.variable = st.sidebar.selectbox(
            "Variables:", sorted(list(self.dataset.variables.keys()))
        )
        st.header(f"**{self.variable} variable**")

    def show_variable_attributes(self):
        """
        This method renders a table which contains the variable 's attributes

        Raises
        ------
        ValueError
            When the variable hasn't been choosen yet.

        Returns
        -------
        None.

        """

        if hasattr(self, "variable"):
            attr_dict = {}
            for attr in self.dataset.variables[self.variable].attrs:
                attr_dict[attr] = [eval(f"self.dataset[self.variable].{attr}")]
            st.subheader("**Variable 's attributes**")
            st.table(pd.DataFrame(data=attr_dict))
        else:
            raise ValueError("The variable hasn't been choosen yet")

    def plot_variable(self):
        """
        PLot the variable

        Returns
        -------
        None.

        """
        st.subheader("**Plots**")
        st.markdown(
            f"*Shape of {self.variable}: {self.dataset.variables[self.variable].shape}*"
        )
        # The variable has only 1 dimension
        if len(self.dataset.variables[self.variable][:].shape) == 1:
            # choose the x absciss from the set of other variables.
            axe = st.selectbox(
                "X axis",
                [
                    var
                    for var in self.dataset.variables
                    if len(self.dataset[var].shape) == 1 and var != self.variable
                ],
            )
            # If the x absciss variable has not the same shape of
            # the variable to be ploted
            if (
                self.dataset.variables[axe][:].shape
                != self.dataset.variables[self.variable][:].shape
            ):
                st.error(
                    f"{axe} and {self.variable} must have same first dimension, but have shapes {self.dataset.variables[axe].shape} and {self.dataset.variables[self.variable][:].shape}"
                )
                # Stop flow
                st.stop()
            # plot the variable
            fig = go.Figure()
            # Create and style traces
            fig.add_trace(
                go.Scatter(
                    x=self.dataset.variables[axe],
                    y=self.dataset.variables[self.variable],
                    name=self.variable,
                    line=dict(color="black", width=2.8, dash="dot"),
                )
            )
            # Check units
            if hasattr(self.dataset.variables[axe], "units"):
                x_unit = self.dataset.variables[axe].units
            elif hasattr(self.dataset.variables[axe], "unit"):
                x_unit = self.dataset.variables[axe].unit
            else:
                x_unit = None
            if hasattr(self.dataset.variables[self.variable], "units"):
                y_unit = self.dataset.variables[self.variable].units
            elif hasattr(self.dataset.variables[self.variable], "unit"):
                y_unit = self.dataset.variables[self.variable].unit
            else:
                y_unit = None
            # layout
            fig.update_layout(
                title=self.variable,
                xaxis_title=f"{str(axe)} ({x_unit})"
                if x_unit is not None
                else str(axe),
                yaxis_title=f"{self.variable} ({y_unit})"
                if y_unit is not None
                else f"{self.variable}",
                plot_bgcolor=None,
            )
            st.plotly_chart(fig)
        # The variable has more than 1 dimension
        else:
            if 1 in self.dataset[self.variable].shape:
                where_1d = self.dataset[self.variable].shape.index(1)
                if where_1d == 0:
                    self.dataset[self.variable] = self.dataset[self.variable][0]
                elif where_1d == 1:
                    self.dataset[self.variable] = self.dataset[self.variable][:, 0, :]
                else:
                    self.dataset[self.variable] = self.dataset[self.variable][:, :, 0]

            with st.beta_container():
                # 2D variable
                if (
                    len(self.dataset.variables[self.variable].shape) == 2
                    or 1 in self.dataset.variables[self.variable].shape
                ):
                    # Default cmap
                    cmap = "jet"
                    if st.sidebar.checkbox("Choose cmap"):
                        # Choose a cmap
                        type_cmap = st.sidebar.radio(
                            "cmap types", self.configfile["default"]["cmap_types"]
                        )
                        cmap = st.sidebar.selectbox(
                            "cmap",
                            [
                                c
                                for c in dir(eval(f"plotly_express.colors.{type_cmap}"))
                                if c[0] != "_"
                            ],
                        )
                    fig = plotly_express.imshow(
                        self.dataset[self.variable],
                        color_continuous_scale=cmap,
                        origin="lower",
                        title=f"2D plot of {self.variable}",
                    )
                    st.plotly_chart(fig)
                # 3D variable
                elif len(self.dataset.variables[self.variable].shape) == 3:
                    # Choose a translation axis from these 3 dimensions
                    dim = st.radio(
                        "Choose a translation axis from these 3 dimensions: ",
                        list(self.dataset[self.variable].dims),
                        key="Fix indentation 4 2d plot",
                    )
                    # Choose the value of the {dim} dimension
                    dim_value = st.select_slider(
                        f"Choose the value of the {dim} dimension",
                        options=self.dataset[dim].data.tolist(),
                        key="plot slider",
                    )
                    # Get the 2d section of the dataset
                    dataset = eval(
                        f"self.dataset[self.variable].loc[dict({dim}=dim_value)]"
                    )

                    cmap = "jet"
                    if st.sidebar.checkbox("Choose cmap"):
                        type_cmap = st.sidebar.radio(
                            "cmap type", self.configfile["default"]["cmap_types"]
                        )
                        cmap = st.sidebar.selectbox(
                            "cmap",
                            [
                                c
                                for c in dir(eval(f"plotly_express.colors.{type_cmap}"))
                                if c[0] != "_"
                            ],
                        )
                    # Check units
                    if hasattr(self.dataset[dim], "units"):
                        dim_unit = self.dataset[dim].units
                    elif hasattr(self.dataset[dim], "unit"):
                        dim_unit = self.dataset[dim].unit
                    else:
                        dim_unit = ""
                    fig = plotly_express.imshow(
                        dataset,
                        color_continuous_scale=cmap,
                        origin="lower",
                        title=f"2D plot of {self.variable} in {dim} = {dim_value} {dim_unit}",
                    )
                    fig.update_layout(plot_bgcolor="white")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("The maximum allowed dataset dimension is 3 dimension")
                    st.stop()

    def stats(self):
        """
        Render the statistics of the variable

        Returns
        -------
        None.

        """

        st.subheader("**Statistics**")
        st.markdown(
            f"*Shape of {self.variable}: {self.dataset.variables[self.variable].shape}*"
        )
        if len(self.dataset[self.variable].shape) == 1:
            # 1d variable
            # Render the statistics in a table
            st.table(
                pd.DataFrame(
                    data={
                        "min": [np.nanmin(self.dataset.variables[self.variable])],
                        "max": [np.nanmax(self.dataset.variables[self.variable])],
                        "mean": [np.nanmean(self.dataset.variables[self.variable])],
                        "std": [np.nanstd(self.dataset.variables[self.variable].data)],
                    }
                )
            )
        else:
            # Define a plot functions
            def plot_stat(variable_name, dataset, stat, axe, axis, **kargs):
                # inverse the axis to compute the statistics along
                # this axis
                axis = 0 if axis else 1
                # default line
                if "line" not in kargs:
                    kargs["line"] = self.configfile["default"]["lines"]

                fig = go.Figure()
                # Create and style traces
                fig.add_trace(
                    go.Scatter(
                        x=self.dataset.variables[axe],
                        y=self.dict_funcs[stat](dataset, axis),
                        name=f"{stat} {variable_name}",
                        line=kargs["line"],
                    )
                )
                # check units
                if hasattr(self.dataset.variables[axe], "units"):
                    x_unit = self.dataset.variables[axe].units
                elif hasattr(self.dataset.variables[axe], "unit"):
                    x_unit = self.dataset.variables[axe].unit
                else:
                    x_unit = None
                if hasattr(dataset, "units"):
                    y_unit = dataset.units
                elif hasattr(dataset, "unit"):
                    y_unit = dataset.unit
                else:
                    y_unit = None
                fig.update_layout(
                    title=f"{stat} sla",
                    xaxis_title=f"{str(axe)} ({x_unit})"
                    if x_unit is not None
                    else str(axe),
                    yaxis_title=f"{stat} {self.variable} ({y_unit})"
                    if y_unit is not None
                    else f"{stat} {self.variable}",
                    plot_bgcolor=None,
                )
                st.plotly_chart(fig)

            if len(self.dataset.variables[self.variable].shape) == 2:
                # 2d variable
                # get the dimensions
                twod_dims = list(self.dataset[self.variable].dims)
                # choose one dimension to compute the statistics along it
                stat_axis = st.sidebar.radio("Statistics direction: ", twod_dims)
                stat_axis = twod_dims.index(stat_axis)
                # chose a x absciss variable
                axe = st.selectbox(
                    "X axis",
                    [
                        var
                        for var in self.dataset.variables
                        if len(self.dataset[var].shape) == 1 and var != self.variable
                    ],
                )
                if (
                    self.dataset.variables[axe].shape[0]
                    != self.dataset[self.variable].shape[stat_axis]
                ):
                    st.error(
                        f"PLease choose an absice which his lenght equal to the number of the time series of {self.variable} variable"
                    )
                    st.stop()

                if st.checkbox("min"):
                    plot_stat(
                        self.variable,
                        self.dataset[self.variable],
                        "min",
                        axe,
                        stat_axis,
                    )

                if st.checkbox("max"):
                    plot_stat(
                        self.variable,
                        self.dataset[self.variable],
                        "max",
                        axe,
                        stat_axis,
                    )

                if st.checkbox("mean"):
                    plot_stat(
                        self.variable,
                        self.dataset[self.variable],
                        "mean",
                        axe,
                        stat_axis,
                    )

                if st.checkbox("std"):
                    plot_stat(
                        self.variable,
                        self.dataset[self.variable],
                        "std",
                        axe,
                        stat_axis,
                    )

            elif len(self.dataset.variables[self.variable].shape) == 3:
                # 3d variable
                dim = st.radio(
                    "Choose a translation axis from these 3 dimensions:",
                    list(self.dataset[self.variable].dims),
                    key="Choose a translation axis from these 3 dimensions 4 stats",
                )
                dim_value = st.select_slider(
                    f"Choose the value of the {dim} dimension",
                    options=self.dataset[dim].data.tolist(),
                    key="stat slider",
                )
                # dataset = eval(f"self.dataset[self.variable].isel({dim}=dim_index)")
                dataset = eval(
                    f"self.dataset[self.variable].loc[dict({dim}=dim_value)]"
                )
                twod_dims = [i for i in self.dataset[self.variable].dims if i != dim]
                stat_axis = st.sidebar.radio("Statistics direction: ", twod_dims)
                stat_axis = twod_dims.index(stat_axis)
                axe = st.selectbox(
                    "X axis",
                    [
                        var
                        for var in self.dataset.variables
                        if len(self.dataset[var].shape) == 1 and var != self.variable
                    ],
                )
                if self.dataset.variables[axe].shape[0] != dataset.shape[stat_axis]:
                    st.error(f"Shape mismatch")
                    st.stop()
                if st.checkbox("min"):
                    plot_stat(self.variable, dataset, "min", axe, stat_axis)
                if st.checkbox("max"):
                    plot_stat(self.variable, dataset, "max", axe, stat_axis)
                if st.checkbox("mean"):
                    plot_stat(self.variable, dataset, "mean", axe, stat_axis)
                if st.checkbox("std"):
                    plot_stat(self.variable, dataset, "std", axe, stat_axis)
            else:
                st.error("The maximum allowed dataset dimension is 3 dimension")
                st.stop()

    def close_file(self):
        """
        Close the file

        Returns
        -------
        None.

        """
        self.dataset.close()


def main():
    """
    Main function which excecutes all the process

    Returns
    -------
    None.

    """
    # General presentation
    Ncpyviewer.general_presentation()
    # Create an instance object of the class object Ncpyviewer
    ncviewer = Ncpyviewer.open_dataset()
    # Read config file
    ncviewer.read_config()
    if st.sidebar.checkbox("Show file 's attributes"):
        ncviewer.show_file_attributes()
    ncviewer.select_variable()
    if st.sidebar.checkbox("Show variable 's attributes"):
        ncviewer.show_variable_attributes()
    if st.sidebar.checkbox("Plot"):
        ncviewer.plot_variable()
    if st.sidebar.checkbox("Statistics"):
        ncviewer.stats()
    ncviewer.close_file()


if __name__ == "__main__":
    main()
