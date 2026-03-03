"""Streamlit utils."""

import logging

import streamlit as st


class StreamlitLogger:
    """Pickup logger and write to Streamlit front end."""

    def __init__(self, placeholder, logger_name=None, accumulate=True, persist=True):
        """
        Pickup logger and write to Streamlit front end.

        Parameters
        ----------
            placeholder: streamlit.empty
                Streamlit placeholder object on which to write logs.
            logger_name: str, optional
                Module name of logger to pick up. Leave to None to pick up root logger.
            accumulate: boolean, optional
                Whether to accumulate log messages or to overwrite with each new
                message, keeping only the last line. (default: True)
            persist: boolean, optional
                Wheter to keep the log when finished, or empty the placeholder element.

        """
        self.placeholder = placeholder
        self.persist = persist

        self.logging_stream = _StreamlitLoggingStream(placeholder, accumulate)
        self.handler = logging.StreamHandler(self.logging_stream)
        self.logger = logging.getLogger(logger_name)

    def __enter__(self):
        self.logger.addHandler(self.handler)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.removeHandler(self.handler)
        if not self.persist:
            self.placeholder.empty()


class _StreamlitLoggingStream:
    """Logging stream that writes logs to Streamlit front end."""

    def __init__(self, placeholder, accumulate=True):
        """
        Logging stream that writes logs to Streamlit front end.

        Parameters
        ----------
            placeholder: streamlit.empty
                Streamlit placeholder object on which to write logs
            accumulate: boolean, optional
                Whether to accumulate log messages or to overwrite with each new
                message, keeping only the last line (default: True)
        """
        self.placeholder = placeholder
        self.accumulate = accumulate
        self.message_list = []

    def write(self, message):
        if self.accumulate:
            self.message_list.append(message)
        else:
            self.message_list = [message]
        self.placeholder.markdown("```\n" + "".join(self.message_list) + "\n```")


def hide_streamlit_menu():
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)


@st.cache
def save_dataframe(df):
    """Save dataframe to file object, with streamlit cache."""
    return df.to_csv().encode("utf-8")
