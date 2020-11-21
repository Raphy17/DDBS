from bokeh.io import output_file, show, save
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Line, HoverTool, FactorRange
import bokeh.palettes as bp

def draw_partitions(S, T, parts):
    p = figure(plot_width=1000, plot_height=600)
    count = 1
    colors = bp.viridis(100)
    col = 0
    for el in parts:
        start_x = []
        end_x = []
        start_y = []
        end_y = []
        width = []
        height = []
        parts = []

        for part in el:
            parts.append(part)
            partition = part.get_A()
            sx = partition[0][0]
            ex = partition[0][1]
            sy = partition[1][0]
            ey = partition[1][1]
            w = abs(ex - sx)
            h = abs(ey - sy)
            start_x.append(sx)
            end_x.append(ex)
            start_y.append(sy)
            end_y.append(ey)
            width.append(w)
            height.append(h)
            col += 1

        center_x = [(x1 + x2) / 2 for x1, x2 in zip(end_x, start_x)]
        center_y = [(y1 + y2) / 2 for y1, y2 in zip(end_y, start_y)]

        part_names = []
        for i in range(len(center_x)):
            part_names.append("P{}".format(count))


        p.rect(x=center_x[-1], y=center_y[-1], width=width[-1],
               height=height[i], fill_color=colors[col], line_color=colors[col], legend_label=part_names[i],
               name=part_names[i], visible=False)

        part_sub = parts[-1]
        sub_x = []
        sub_y = []
        if not part_sub.regular_partition:
            subs = part_sub.sub_partitions

            for j in range(1, subs[0]):
                sub_x.append(start_x[-1]+width[-1]/subs[0]*j)

            for j in range(1, subs[1]):
                sub_y.append(start_y[-1]+height[-1]/subs[1]*j)

        for e in sub_x:
            p.line(x=(e,e), y=(start_y[-1],end_y[-1]), line_dash='dashed', line_color="black")

        for l in sub_y:
            p.line(x=(start_x[-1],end_x[-1]), y=(l,l), line_dash='dashed', line_color="black")


        count += 1
        p.legend.click_policy = "hide"

    hover = HoverTool(tooltips=[("name", "$name"), ("x", "$x"), ("y", "$y")])

    p.add_tools(hover)

    for i in range(len(S)):
        p.cross(x=S[i][0], y=S[i][1], line_color="blue")

    for i in range(len(T)):
        p.cross(x=T[i][0], y=T[i][1], line_color="black")

    show(p)


def draw_samples(S, T):
    p = figure(plot_width=1000, plot_height=600)
    count = 1
    for i in range(len(S)):
        p.cross(x=S[i][0], y=S[i][1], line_color="blue")

    for i in range(len(T)):
        p.cross(x=T[i][0], y=T[i][1], line_color="black")

    #show(p)