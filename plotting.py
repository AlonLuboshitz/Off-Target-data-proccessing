import matplotlib.pyplot as plt



'''given a x data and y data draws a auc curve.
add marking lines to x_pinpoint and y_points - i.a y = 0.5 draw a line there to mark this value
y_data is a dictionary - key: (expirement name\guide rna), value: ascending rates
plot all keys through the x data
x data is a dictionary with one key: name of label, value: data '''
def draw_auc_curve(x_data,y_data,x_pinpoints,y_pinpoints,title):
    # Plot the AUC curve
    plt.figure()

    for key_x,ranks in x_data.items():   
        for key_y,rates in y_data.items():
            plt.plot(ranks, rates,label=key_y)
    plt.xlabel(key_x)
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.show()
