"""Small single-artifact notebook viewers; no inference or state changes."""
from html import escape
from typing import Mapping, Sequence


def format_cell_value(value: object) -> str:
    """Format a scalar or short vector for display without changing stored data.

    Parameters
    ----------
    value : object
        Scalar, list or tuple; floats are shown with six significant digits.
        Nested lists preserve their row boundaries. None denotes a missing value.

    Returns
    -------
    str
        Display text only; original numerical values remain unchanged.

    Raises
    ------
    ValueError
        If a mapping is supplied as one cell. Flatten named metrics into columns
        or separate tables instead of displaying a dense nested result object.
    """
    if value is None:
        return '—'
    if isinstance(value, Mapping):
        raise ValueError('A table cell cannot contain a result dictionary: use named scalar columns')
    if isinstance(value, float):
        return format(value, '.6g')
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(format_cell_value(item) for item in value) + ']'
    return str(value)


def show_table(
    rows: Sequence[Mapping[str, object]], columns: Sequence[str], caption: str,
) -> None:
    """Display one labelled table with safely escaped cell values.

    Parameters
    ----------
    rows : Sequence[Mapping[str, object]]
        Records for the named columns; missing/None values display an em dash.
        Nested result dictionaries must be decomposed. Display rounding does not
        change the recorded values or calculations.
        An empty population displays an empty table with its caption.
    columns : Sequence[str]
        Display order of fields; headers should carry units where relevant.
    caption : str
        Question/population/evidence scope for this single table.

    Returns
    -------
    None
        Displays one HTML artifact. Does not execute inference or save files.
    """
    from IPython.display import HTML, display
    headers = ''.join('<th scope="col" style="text-align:left;padding:.4em .6em">'
                      + escape(name.replace('_', ' ')) + '&#160;&#160;</th>' for name in columns)
    body = ''.join('<tr>' + ''.join('<td style="vertical-align:top;padding:.4em .6em;'
                    'max-width:32em;overflow-wrap:anywhere;'
                    'border-bottom:1px solid #ccc">' + escape(format_cell_value(row.get(key))) + '&#160;&#160;</td>'
                    for key in columns) + '</tr>' for row in rows)
    display(HTML('<div style="max-width:100%;overflow-x:auto"><table style="border-collapse:collapse">'
                 '<caption style="text-align:left">' + escape(caption)
                 + '<br><small>Display: six significant digits; stored values unchanged.</small>'
                 '</caption><thead><tr>' + headers + '</tr></thead><tbody>'
                 + body + '</tbody></table></div>'))


def show_text(title: str, text: str) -> None:
    """Display one literal input/output or diagnostic text artifact.

    Parameters
    ----------
    title : str
        Case identity and evidence scope.
    text : str
        Literal content; HTML is escaped, not interpreted.

    Returns
    -------
    None
        One readable text panel; no GPU call, write or state change.
    """
    from IPython.display import HTML, display
    display(HTML('<h4>' + escape(title) + '</h4><pre style="white-space:pre-wrap">'
                 + escape(text) + '</pre>'))
