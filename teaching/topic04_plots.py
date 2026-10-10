"""Readable task figures; no network or GPU work at import."""

def annotation_color(rgba):
    # Relative luminance, with sRGB linearization; choose higher text contrast.
    values=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgba[:3]]
    light=sum(w*v for w,v in zip((.2126,.7152,.0722),values))
    return 'black' if (light+.05)/.05 >= 1.05/(light+.05) else 'white'


def plot_matrix(rows, score, title):
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


def plot_sampling(probs):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,3.4),layout='constrained')
    ax.bar(['token A','token B','token C'],probs)
    ax.set(ylim=(0,1),ylabel='Next-token probability',title='Toy logits [2,1,0], after T and top_p')
    plt.show()
    return fig
