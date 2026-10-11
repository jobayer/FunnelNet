import seaborn as sns


def bar(X, y, title=None, ax=None, pal=None, sd=13):
    n_colors = len(set(X))
    plte = sns.color_palette(pal, n_colors)
    barplot = sns.barplot(x=X, y=y, hue=X, ax=ax, palette=plte, seed=sd)

    if ax is not None:
        ax.grid(True, linewidth=0.5, color="gray", alpha=0.2)
        ax.set_axisbelow(True)

        i = 0

        for p in barplot.patches:

            if i ==4:
                barplot.annotate(
                    format(p.get_height(), ".0f"),
                    (p.get_x() + p.get_width() / 2.0, p.get_height() + 40),
                    ha="center",
                    va="center",
                    fontsize=38,
                    xytext=(0, 10),
                    textcoords="offset points",
                    rotation=90,
                )

            i = i + 1

        ax.set_ylim(0, max(y) * 1.15)

    if title is not None and ax is not None:
        ax.set_title(title)
