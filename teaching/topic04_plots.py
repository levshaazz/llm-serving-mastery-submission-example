"""Readable task figures; no network or GPU work at import."""
from typing import Callable, Sequence, TYPE_CHECKING
from teaching.topic04_types import PredictionRow,PredictionScores
if TYPE_CHECKING:
    from matplotlib.figure import Figure

def annotation_color(rgba: Sequence[float]) -> str:
    """Choose black/white text using linearized sRGB relative luminance.

    Parameters
    ----------
    rgba : sequence of float
        Color channels in [0,1]; alpha is ignored.

    Returns
    -------
    str
        The black or white color with greater contrast against RGB.
    """
    # Relative luminance, with sRGB linearization; choose higher text contrast.
    values=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgba[:3]]
    light=sum(w*v for w,v in zip((.2126,.7152,.0722),values))
    return 'black' if (light+.05)/.05 >= 1.05/(light+.05) else 'white'


def plot_matrix(rows: Sequence[PredictionRow], score: Callable[[Sequence[PredictionRow]], PredictionScores], title: str) -> 'Figure':
    """Display one all-attempt confusion matrix, including invalid predictions.

    Parameters
    ----------
    rows : sequence of PredictionRow
        Every attempted ticket, including incomplete/failed responses.
    score : callable
        Returns a [3,4] true-row/predicted-column count matrix.
    title : str
        Figure title identifying population and evidence type.

    Returns
    -------
    Figure
        Displayed figure; counts include all attempts, colors auto-scale.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    labels=('billing','access','bug')
    matrix=np.array(score(rows)['confusion'])
    fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained')
    im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(1,matrix.max()))
    for i in range(3):
        for j in range(4):
            ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=13,
                    color=annotation_color(im.cmap(im.norm(matrix[i,j]))))
    ax.set(xticks=range(4),xticklabels=[*labels,'invalid /\nincomplete'],yticks=range(3),
           yticklabels=labels,xlabel='Predicted label',ylabel='True label',title=title)
    fig.colorbar(im,ax=ax,label='Attempt count',ticks=range(int(matrix.max())+1))
    plt.show()
    return fig


def plot_sampling(probs: Sequence[float]) -> 'Figure':
    """Display one invented three-token sampling distribution.

    Parameters
    ----------
    probs : sequence of float
        Three probabilities in vocabulary order, not class accuracy scores.

    Returns
    -------
    Figure
        Displayed bar chart with probability axis fixed to [0,1].
    """
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,3.4),layout='constrained')
    ax.bar(['token A','token B','token C'],probs)
    ax.set(ylim=(0,1),ylabel='Next-token probability',title='Toy logits [2,1,0], after T and top_p')
    plt.show()
    return fig
